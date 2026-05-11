// subprocess.cpp — implementation of yvm/subprocess.h
#include "subprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <thread>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <chrono>

namespace yvm {

bool probe_video(const std::string& path, int& w, int& h, double& fps, int64_t& nframes) {
    char cmd[1024];
    std::snprintf(cmd, sizeof cmd,
        "ffprobe -v error -select_streams v:0 -show_entries "
        "stream=width,height,r_frame_rate,nb_frames -of default=nw=1 '%s'",
        path.c_str());
    FILE* p = popen(cmd, "r");
    if (!p) { std::perror("popen ffprobe"); return false; }
    char line[256];
    w = 0; h = 0; fps = 0; nframes = 0;
    while (std::fgets(line, sizeof line, p)) {
        int a, b;
        if      (std::sscanf(line, "width=%d", &a) == 1)               w = a;
        else if (std::sscanf(line, "height=%d", &a) == 1)              h = a;
        else if (std::sscanf(line, "r_frame_rate=%d/%d", &a, &b) == 2 && b) fps = (double)a / b;
        else if (std::sscanf(line, "nb_frames=%lld", (long long*)&nframes) == 1) {}
    }
    pclose(p);
    return w > 0 && h > 0;
}

Subprocess ffmpeg_reader(const std::string& path, int& w, int& h, double& fps,
                         int64_t& nframes, int target_fps)
{
    if (!probe_video(path, w, h, fps, nframes)) std::exit(1);

    int pipefd[2];
    if (pipe(pipefd) < 0) { std::perror("pipe"); std::exit(1); }
    pid_t pid = fork();
    if (pid == 0) {
        ::close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        ::close(pipefd[1]);
        char rate_arg[32];
        std::snprintf(rate_arg, sizeof rate_arg, "%d", target_fps);
        execlp("ffmpeg",
               "ffmpeg", "-loglevel", "error",
               "-stream_loop", "-1",
               "-re",                                  // pace input at native rate
               "-i", path.c_str(),
               "-r", rate_arg,                         // re-clock to target_fps
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
               (char*)nullptr);
        _exit(127);
    }
    ::close(pipefd[1]);
    return Subprocess{pid, pipefd[0]};
}

Subprocess ffmpeg_writer(const std::string& path, int w, int h, double fps) {
    int pipefd[2];
    if (pipe(pipefd) < 0) { std::perror("pipe"); std::exit(1); }
    pid_t pid = fork();
    if (pid == 0) {
        ::close(pipefd[1]);
        dup2(pipefd[0], STDIN_FILENO);
        ::close(pipefd[0]);
        char size[32], rate[32];
        std::snprintf(size, sizeof size, "%dx%d", w, h);
        std::snprintf(rate, sizeof rate, "%.6f", fps);
        execlp("ffmpeg",
               "ffmpeg", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", size, "-r", rate, "-i", "-",
               "-c:v", "h264_rkmpp", "-pix_fmt", "nv12",
               path.c_str(),
               (char*)nullptr);
        _exit(127);
    }
    ::close(pipefd[0]);
    return Subprocess{pid, pipefd[1]};
}

Subprocess gst_local_display(int w, int h, double fps, const char* /*title*/) {
    int pipefd[2];
    if (pipe(pipefd) < 0) { std::perror("pipe"); std::exit(1); }
    pid_t pid = fork();
    if (pid == 0) {
        ::close(pipefd[1]);
        dup2(pipefd[0], STDIN_FILENO);
        ::close(pipefd[0]);
        char w_arg[32], h_arg[32], r_arg[32];
        std::snprintf(w_arg, sizeof w_arg, "width=%d", w);
        std::snprintf(h_arg, sizeof h_arg, "height=%d", h);
        std::snprintf(r_arg, sizeof r_arg, "framerate=%d/1", (int)std::round(fps));
        const char* argv_gst[] = {
            "gst-launch-1.0", "-q",
            "fdsrc", "fd=0", "!",
            "rawvideoparse", "format=bgr", w_arg, h_arg, r_arg, "!",
            "queue", "max-size-buffers=2", "leaky=downstream", "!",
            "videoconvert", "!",
            "autovideosink", "sync=false",
            nullptr
        };
        execvp(argv_gst[0], (char* const*)argv_gst);
        _exit(127);
    }
    ::close(pipefd[0]);
    return Subprocess{pid, pipefd[1]};
}

Subprocess ffmpeg_tcp_streamer(int w, int h, double fps, int port) {
    int pipefd[2];
    if (pipe(pipefd) < 0) { std::perror("pipe"); std::exit(1); }
    pid_t pid = fork();
    if (pid == 0) {
        ::close(pipefd[1]);
        dup2(pipefd[0], STDIN_FILENO);
        ::close(pipefd[0]);
        char size[32], rate[32], url[64];
        std::snprintf(size, sizeof size, "%dx%d", w, h);
        std::snprintf(rate, sizeof rate, "%.6f", fps);
        std::snprintf(url, sizeof url, "tcp://0.0.0.0:%d?listen=1", port);
        execlp("ffmpeg",
               "ffmpeg", "-loglevel", "warning",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", size, "-r", rate, "-i", "-",
               "-c:v", "h264_rkmpp", "-pix_fmt", "nv12",
               "-f", "mpegts", url,
               (char*)nullptr);
        _exit(127);
    }
    ::close(pipefd[0]);
    return Subprocess{pid, pipefd[1]};
}

bool read_full(int fd, void* buf, size_t n) {
    auto* p = (uint8_t*)buf;
    while (n) {
        ssize_t r = ::read(fd, p, n);
        if (r <= 0) return false;
        p += r; n -= r;
    }
    return true;
}

bool write_full(int fd, const void* buf, size_t n) {
    auto* p = (const uint8_t*)buf;
    while (n) {
        ssize_t r = ::write(fd, p, n);
        if (r <= 0) return false;
        p += r; n -= r;
    }
    return true;
}

void close_sub(Subprocess& s) {
    if (s.fd >= 0) { ::close(s.fd); s.fd = -1; }
    if (s.pid <= 0) return;
    int st = 0;
    for (int i = 0; i < 20; ++i) {                // up to 2.0 s
        pid_t r = waitpid(s.pid, &st, WNOHANG);
        if (r == s.pid) { s.pid = -1; return; }
        if (r < 0)      break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ::kill(s.pid, SIGTERM);
    for (int i = 0; i < 10; ++i) {                // up to 1.0 s
        pid_t r = waitpid(s.pid, &st, WNOHANG);
        if (r == s.pid) { s.pid = -1; return; }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ::kill(s.pid, SIGKILL);
    waitpid(s.pid, &st, 0);
    s.pid = -1;
}

}  // namespace yvm
