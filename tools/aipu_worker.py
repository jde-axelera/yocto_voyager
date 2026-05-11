#!/usr/bin/env python3
"""
aipu_worker.py — persistent AIPU dispatcher used by yolo_demo_multi as a Python
side-car. The C++ binary spawns this process, hands it the model path plus the
input/output dma-buf fds over a Unix socket (via SCM_RIGHTS), then signals one
byte per batch to trigger `instance.run`.

Why a side-car instead of inline `axr_run_model_instance`:
  The optimized AIPU dispatch path lives behind `libruntime2_core.so` which
  exports zero dynamic symbols and is only reachable via the bundled
  `_core.cpython-310-aarch64-linux-gnu.so` Python extension. Calling the public
  C API from C++ caps us at ~240 system fps on yolo11n; routing the dispatch
  through the Python `axelera.runtime` package matches `axrunmodel`'s
  ~425–540 system fps ceiling because the runtime can internally pipeline.

Protocol (Unix socket, abstract namespace, length-prefixed setup):
    C → P :  4-byte BE length, then JSON setup payload:
              { "model_path": "...",
                "batch":        4,
                "aipu_cores":   4,
                "n_out":        6,
                "double_buffer": true,
                "output_dmabuf": false }
              ancillary data carries (input_fd, [output_fds...]) via SCM_RIGHTS.
              If "output_dmabuf" is false, only input_fd is passed; outputs are
              allocated host-side here and discarded by the C++ caller.
    P → C :  1 byte "K" on ready, "E" on error.

Per batch:
    C → P :  one byte "G".
    P → C :  one byte "D" on success, "E" on error.

To stop: C closes the socket.
"""
from __future__ import annotations

import array
import json
import logging
import os
import socket
import struct
import sys
import time
import traceback

import numpy as np

from axelera.runtime.objects import Context

LOG = logging.getLogger("aipu_worker")


def _recv_fds(sock: socket.socket, payload_len: int, max_fds: int):
    """recvmsg with SCM_RIGHTS, return (payload_bytes, [fds])."""
    fds = array.array("i")
    msg, ancdata, _, _ = sock.recvmsg(
        payload_len, socket.CMSG_SPACE(max_fds * fds.itemsize)
    )
    fds_out: list[int] = []
    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fds.frombytes(cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])
            fds_out.extend(fds.tolist())
    return msg, fds_out


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    while n > 0:
        b = sock.recv(n)
        if not b:
            raise EOFError("peer closed")
        chunks.append(b)
        n -= len(b)
    return b"".join(chunks)


def main(socket_path: str) -> int:
    logging.basicConfig(
        level=os.environ.get("AIPU_WORKER_LOG", "WARNING"),
        format="[aipu_worker] %(levelname)s %(message)s",
    )

    # Bind a SOCK_STREAM UNIX socket and wait for the one connection from C++.
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    srv.bind(socket_path)
    srv.listen(1)
    LOG.info("listening on %s", socket_path)
    conn, _ = srv.accept()
    LOG.info("client connected")

    # ---- setup phase ----
    # 4-byte BE length prefix, then payload + ancillary fds in the same recvmsg.
    hdr = _recv_exact(conn, 4)
    payload_len = struct.unpack(">I", hdr)[0]
    payload, fds = _recv_fds(conn, payload_len, max_fds=16)
    if len(payload) != payload_len:
        LOG.error("short setup payload")
        conn.sendall(b"E")
        return 1
    cfg = json.loads(payload.decode("utf-8"))
    LOG.info("setup cfg=%s fds=%s", cfg, fds)

    model_path    = cfg["model_path"]
    batch_size    = int(cfg["batch"])
    aipu_cores    = int(cfg["aipu_cores"])
    n_out         = int(cfg["n_out"])
    double_buffer = bool(cfg.get("double_buffer", True))
    output_dmabuf = bool(cfg.get("output_dmabuf", False))

    if output_dmabuf:
        if len(fds) != 1 + n_out:
            LOG.error("expected %d fds with output_dmabuf=1, got %d", 1 + n_out, len(fds))
            conn.sendall(b"E")
            return 1
    else:
        if len(fds) != 1:
            LOG.error("expected 1 fd (input only), got %d", len(fds))
            conn.sendall(b"E")
            return 1

    input_fd  = fds[0]
    output_fds = fds[1:] if output_dmabuf else []

    try:
        ctx = Context()
        model = ctx.load_model(str(model_path))
        devices = ctx.list_devices()
        if not devices:
            raise RuntimeError("no devices available")
        dev = devices[0]
        ctx.configure_device(dev, device_firmware='1')
        ax_conn = ctx.device_connect(dev, batch_size, device_firmware_check=0)
        instance = ax_conn.load_model_instance(
            model,
            double_buffer=double_buffer,
            input_dmabuf=True,
            output_dmabuf=output_dmabuf,
            num_sub_devices=batch_size,
            aipu_cores=aipu_cores,
        )
        output_infos = model.outputs()
        if output_dmabuf:
            outputs_arg: list = list(output_fds)
        else:
            # Heap output: allocate one batch-shaped numpy array per output tensor.
            # These are reused across all calls.
            outputs_arg = [np.zeros(t.shape, np.int8) for t in output_infos]
    except Exception:
        LOG.error("setup failed:\n%s", traceback.format_exc())
        conn.sendall(b"E")
        return 1

    inputs_arg = [input_fd]

    conn.sendall(b"K")
    LOG.info("ready; entering inference loop")

    # ---- per-batch RPC ----
    # Tight loop; one byte in, one byte out.  Anything else is treated as quit.
    n_calls = 0
    t0 = time.time()
    while True:
        b = conn.recv(1)
        if not b:
            break
        if b == b"G":
            try:
                instance.run(inputs_arg, outputs_arg)
                conn.sendall(b"D")
                n_calls += 1
            except Exception:
                LOG.error("run failed:\n%s", traceback.format_exc())
                conn.sendall(b"E")
        else:
            break

    dt = time.time() - t0
    if dt > 0 and n_calls:
        LOG.info("served %d batches in %.2fs (~%.1f batch/s)",
                 n_calls, dt, n_calls / dt)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: aipu_worker.py <socket-path>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
