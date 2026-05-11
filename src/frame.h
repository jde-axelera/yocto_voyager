// frame.h
//
// Per-frame and per-stream state structs threaded through the pipeline.
//
// One `Frame` is built by the decoder (raw BGR), enriched by the preprocessor
// (int8 input plus axrArgument metadata), filled by the worker (raw int8
// outputs sliced from a batch inference), then consumed by the drawer +
// per-stream writer.
//
// A `Stream` is the long-lived per-input state: ffmpeg subprocesses, output
// queue + writer thread, per-stream reorder buffer used by the drawer, the
// most-recent annotated frame snapshot used by the composite display, and a
// few atomic counters for live stats.
#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "axruntime/axruntime.h"
#include "subprocess.h"
#include "concurrency.h"

namespace yvm {

struct Frame {
    int  idx       = -1;             ///< per-stream frame index (used for reorder)
    int  stream_id = 0;
    int  sw = 0, sh = 0;

    std::vector<uint8_t> bgr;        ///< raw BGR24 frame (sw*sh*3 bytes)
    std::vector<int8_t>  input;      ///< preprocessed model input (one batch slot)
    std::vector<std::vector<int8_t>> outputs;  ///< one buffer per output tensor (per-slot bytes)

    std::vector<axrArgument> in_args, out_args;

    // Letterbox transform from preprocess; needed for post-decode coord undo.
    float lscale = 1.0f;
    int   padx = 0, pady = 0;
};

using FramePtr = std::unique_ptr<Frame>;

struct Stream {
    int          id;
    std::string  in_path;
    std::string  out_path;
    int          sw = 0, sh = 0;
    double       fps_in = 30.0;
    int64_t      nframes = 0;

    Subprocess   reader{}, writer{};
    BoundedQueue<FramePtr>* write_q = nullptr;
    std::thread  writer_thread;

    // Per-stream reorder state, used inside the drawer thread to emit frames
    // in increasing idx even if the upstream pipeline reordered them.
    std::map<int, FramePtr> pending;
    int next_idx = 0;

    // Live counters (atomic so the stats logger can sample without locks).
    std::atomic<int> decoded{0};
    std::atomic<int> drawn{0};
    std::atomic<int> written{0};

    // Most-recent annotated frame for the composite display thread.
    std::mutex             snap_mu;
    std::vector<uint8_t>   snapshot;
};

}  // namespace yvm
