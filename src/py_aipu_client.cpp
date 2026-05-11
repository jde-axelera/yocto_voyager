// py_aipu_client.cpp — implementation of yvm/py_aipu_client.h.
//
// The protocol is described in tools/aipu_worker.py. Briefly:
//   setup : C → P : 4-byte BE length + JSON payload + SCM_RIGHTS fds
//   setup : P → C : 1 byte "K" ready / "E" error
//   per-batch : C → P "G" / P → C "D" or "E"
#include "py_aipu_client.h"

#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <signal.h>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

namespace yvm {

namespace {

bool send_all(int fd, const void* buf, size_t n) {
    const uint8_t* p = (const uint8_t*)buf;
    while (n) {
        ssize_t r = ::send(fd, p, n, 0);
        if (r <= 0) return false;
        p += r; n -= r;
    }
    return true;
}

bool recv_all(int fd, void* buf, size_t n) {
    uint8_t* p = (uint8_t*)buf;
    while (n) {
        ssize_t r = ::recv(fd, p, n, 0);
        if (r <= 0) return false;
        p += r; n -= r;
    }
    return true;
}

/// sendmsg with SCM_RIGHTS carrying `nfds` file descriptors.
bool sendmsg_with_fds(int sock, const void* buf, size_t n,
                      const int* fds, size_t nfds) {
    struct iovec iov{};
    iov.iov_base = const_cast<void*>(buf);
    iov.iov_len  = n;

    // cmsghdr + nfds * int payload, with proper alignment.
    char cbuf[CMSG_SPACE(sizeof(int) * 16)] = {0};
    if (nfds > 16) return false;

    struct msghdr msg{};
    msg.msg_iov     = &iov;
    msg.msg_iovlen  = 1;
    msg.msg_control = cbuf;
    msg.msg_controllen = CMSG_SPACE(sizeof(int) * nfds);

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type  = SCM_RIGHTS;
    cmsg->cmsg_len   = CMSG_LEN(sizeof(int) * nfds);
    std::memcpy(CMSG_DATA(cmsg), fds, sizeof(int) * nfds);

    ssize_t r = ::sendmsg(sock, &msg, 0);
    return r == (ssize_t)n;
}

}  // namespace

PyAipuClient::~PyAipuClient() { stop(); }

bool PyAipuClient::start(const std::string& worker_script_path,
                         const std::string& model_path,
                         int batch, int aipu_cores, int n_out,
                         bool output_dmabuf,
                         int input_fd,
                         const std::vector<int>& output_fds)
{
    // Unique socket path per process so multiple binaries don't collide.
    char buf[128];
    std::snprintf(buf, sizeof buf, "/tmp/yvm_aipu_%d.sock", (int)getpid());
    sockpath_ = buf;
    ::unlink(sockpath_.c_str());

    struct sockaddr_un sa{};
    sa.sun_family = AF_UNIX;
    std::strncpy(sa.sun_path, sockpath_.c_str(), sizeof(sa.sun_path) - 1);

    // Fork + exec the worker; the worker is the *server* and binds the socket.
    pid_t pid = fork();
    if (pid < 0) { std::perror("py-aipu: fork"); return false; }
    if (pid == 0) {
        execlp("python3", "python3",
               worker_script_path.c_str(), sockpath_.c_str(),
               (char*)nullptr);
        std::perror("py-aipu: execlp python3");
        _exit(127);
    }
    pid_ = pid;

    // Poll-connect until the worker has called bind()+listen() or we time out.
    sock_ = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (sock_ < 0) { std::perror("py-aipu: client socket"); return false; }
    bool connected = false;
    for (int i = 0; i < 200; ++i) {                       // 50 ms × 200 = 10 s
        if (::connect(sock_, (struct sockaddr*)&sa, sizeof sa) == 0) {
            connected = true; break;
        }
        usleep(50 * 1000);
    }
    if (!connected) {
        std::fprintf(stderr, "py-aipu: connect to %s timed out (worker didn't start?)\n",
                     sockpath_.c_str());
        return false;
    }

    // Build JSON setup payload.
    char json[1024];
    int  jlen = std::snprintf(json, sizeof json,
        "{\"model_path\":\"%s\","
        "\"batch\":%d,"
        "\"aipu_cores\":%d,"
        "\"n_out\":%d,"
        "\"double_buffer\":true,"
        "\"output_dmabuf\":%s}",
        model_path.c_str(), batch, aipu_cores, n_out,
        output_dmabuf ? "true" : "false");
    if (jlen <= 0 || (size_t)jlen >= sizeof json) {
        std::fprintf(stderr, "py-aipu: setup payload too long\n");
        return false;
    }

    // 4-byte BE length prefix (sent as its own message; not in the SCM_RIGHTS msg).
    uint32_t be_len = htonl((uint32_t)jlen);
    if (!send_all(sock_, &be_len, 4)) {
        std::perror("py-aipu: send len"); return false;
    }

    // Payload + fds in the same sendmsg.
    std::vector<int> fds;
    fds.reserve(1 + output_fds.size());
    fds.push_back(input_fd);
    if (output_dmabuf) {
        for (int fd : output_fds) fds.push_back(fd);
    }
    if (!sendmsg_with_fds(sock_, json, (size_t)jlen, fds.data(), fds.size())) {
        std::perror("py-aipu: sendmsg setup"); return false;
    }

    // Wait for ready ack.
    char ack = 0;
    if (!recv_all(sock_, &ack, 1)) {
        std::fprintf(stderr, "py-aipu: no ack from worker (it likely crashed during setup)\n");
        return false;
    }
    if (ack != 'K') {
        std::fprintf(stderr, "py-aipu: worker setup failed (ack='%c')\n", ack);
        return false;
    }
    return true;
}

bool PyAipuClient::run_one() {
    if (sock_ < 0) return false;
    char b = 'G';
    if (!send_all(sock_, &b, 1)) return false;
    if (!recv_all(sock_, &b, 1)) return false;
    return b == 'D';
}

void PyAipuClient::stop() {
    if (sock_ >= 0) { ::close(sock_); sock_ = -1; }
    if (pid_ > 0) {
        int st = 0;
        for (int i = 0; i < 20; ++i) {
            pid_t r = waitpid(pid_, &st, WNOHANG);
            if (r == pid_) { pid_ = -1; break; }
            usleep(100 * 1000);
        }
        if (pid_ > 0) {
            ::kill(pid_, SIGTERM);
            for (int i = 0; i < 10; ++i) {
                pid_t r = waitpid(pid_, &st, WNOHANG);
                if (r == pid_) { pid_ = -1; break; }
                usleep(100 * 1000);
            }
            if (pid_ > 0) { ::kill(pid_, SIGKILL); waitpid(pid_, &st, 0); pid_ = -1; }
        }
    }
    if (!sockpath_.empty()) { ::unlink(sockpath_.c_str()); sockpath_.clear(); }
}

}  // namespace yvm
