// py_aipu_client.h
//
// C++ client for tools/aipu_worker.py — a persistent Python side-car that owns
// the AIPU context and runs `axelera.runtime.ModelInstance.run` on our behalf.
// Routing the dispatch through Python recovers the runtime's internal pipeline
// optimization (~425–540 system fps for yolo11n) which is unreachable from the
// public C API alone (~240 system fps).
//
// Usage:
//   PyAipuClient cli;
//   if (!cli.start("/path/to/model.json",
//                  /*batch=*/4, /*aipu_cores=*/4,
//                  /*n_out=*/6, /*output_dmabuf=*/false,
//                  /*input_fd=*/in_fd,
//                  /*output_fds=*/{})) { /* error */ }
//   // per batch:
//   if (!cli.run_one()) { /* error */ }
//   // shutdown handled by destructor.
#pragma once

#include <string>
#include <vector>

namespace yvm {

class PyAipuClient {
public:
    PyAipuClient() = default;
    ~PyAipuClient();
    PyAipuClient(const PyAipuClient&) = delete;
    PyAipuClient& operator=(const PyAipuClient&) = delete;

    /// Spawn aipu_worker.py and hand over the dma-buf fds + model config.
    /// Returns true once the worker has loaded the model + connected to the
    /// device and signalled ready.
    bool start(const std::string& worker_script_path,
               const std::string& model_path,
               int batch, int aipu_cores, int n_out,
               bool output_dmabuf,
               int input_fd,
               const std::vector<int>& output_fds);

    /// Send one "G" byte, wait for one "D" byte. Returns false on RPC error
    /// or if the Python worker reports failure.
    bool run_one();

    /// Stop the worker cleanly (close socket, waitpid).
    void stop();

private:
    int     sock_  = -1;
    int     pid_   = -1;
    std::string sockpath_;
};

}  // namespace yvm
