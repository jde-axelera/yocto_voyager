// yolo_demo_multi.cpp — top-level orchestrator.
//
// Pipeline (single Axelera context, single batch=4 instance, N inputs):
//
//   stream-decoder × N  →  raw_q
//                            │
//                            ▼
//                    preproc worker × M
//                            │
//                            ▼
//                       inst_q[0]
//                            │
//                            ▼
//          worker (b4 packing → axr_run_model_instance → slice outputs)
//                            │
//                            ▼
//                          done_q
//                            │
//                            ▼
//             drawer (per-stream reorder + postproc + draw)
//                            │
//                ┌───────────┴───────────┐
//                ▼                       ▼
//          write_q[i]             snapshot[i]  ──┐
//                │                                │
//                ▼                                ▼
//        writer thread × N      composite display: producer @ 30 Hz → LeakyOne
//        (h264_rkmpp mp4)                                           │
//                                                                    ▼
//                                                      consumer (X11 / TCP H.264)
//
// All non-trivial logic lives in dedicated modules:
//   subprocess.h/.cpp     ffmpeg/gst-launch wiring + read_full/write_full/close_sub
//   dma_heap.h/.cpp       dma-heap buffer pool + cache sync ioctl wrappers
//   drawing.h/.cpp        BGR draw primitives + class_color palette lookup
//   font.h/.cpp           TTF rasterization via stb_truetype (Liberation Sans Bold)
//   yolo_preproc.h/.cpp   letterbox + quantize into model input layout
//   yolo_postproc.h/.cpp  DFL + sigmoid + class-aware NMS
//   frame.h               Frame and Stream state structs
//   concurrency.h         BoundedQueue and LeakyOne templates
//
// CLI:
//   ./yolo_demo_multi <model.json|.axm> <vid1,vid2,...> <out_prefix>
//                    [conf=0.25] [iou=0.45] [workers=1] [bench=0]
//                    [dmabuf=0]  [preproc=4] [display=0] [fps=25]

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <getopt.h>
#include <glob.h>
#include <linux/input.h>
#include <linux/types.h>
#include <map>
#include <memory>
#include <poll.h>
#include <signal.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "axruntime/axruntime.h"

#include "coco_names.h"
#include "concurrency.h"
#include "dma_heap.h"
#include "drawing.h"
#include "font.h"
#include "frame.h"
#include "py_aipu_client.h"
#include "subprocess.h"
#include "yolo_postproc.h"
#include "yolo_preproc.h"

using yvm::BoundedQueue;
using yvm::LeakyOne;
using yvm::Frame;
using yvm::FramePtr;
using yvm::Stream;
using yvm::Subprocess;
using yvm::FontAtlas;
using yvm::InputBufferPool;
using yvm::Detection;
using yvm::Image;

using clk = std::chrono::steady_clock;

namespace {

// Process-wide shutdown flag; flipped by SIGINT/SIGTERM. Threads poll it.
std::atomic<bool> g_shutdown{false};
// Set only on a *forced* quit (SIGINT/SIGTERM/'q'), never on normal EOF
// completion. Arms the hard-exit watchdog so a wedged pipeline can't freeze.
std::atomic<bool> g_force_quit{false};
void on_sig(int) { g_shutdown.store(true); g_force_quit.store(true); }

// PID of the local gst-launch / TCP-streamer display child, published by the
// display consumer so sig_watcher can SIGTERM it on shutdown — that unblocks a
// disp_consumer wedged in write() to a frozen waylandsink (which closing the
// frame queue cannot interrupt). -1 when no child is running.
std::atomic<pid_t> g_disp_pid{-1};

// Number of input streams, published after argv parse. The forced-quit watchdog
// scales its grace period by this — finalizing N MP4 encoders under full CPU
// load (N up to 40) legitimately takes longer than a couple of seconds.
std::atomic<int> g_stream_count{0};

// SIGKILL every process whose parent is us. Used by the watchdog as a last
// resort so a forced _exit() never leaves orphaned ffmpeg/gst children behind.
// Parses /proc/<pid>/stat; comm (field 2) may contain spaces/parens, so we key
// off the last ')' and read state + ppid after it.
void kill_children() {
    pid_t me = getpid();
    DIR* d = opendir("/proc");
    if (!d) return;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        pid_t p = (pid_t)atoi(e->d_name);
        if (p <= 0 || p == me) continue;
        char path[64];
        std::snprintf(path, sizeof path, "/proc/%d/stat", (int)p);
        FILE* f = fopen(path, "r");
        if (!f) continue;
        char buf[512];
        char* line = fgets(buf, sizeof buf, f);
        fclose(f);
        if (!line) continue;
        char* rp = std::strrchr(line, ')');
        if (!rp) continue;
        char state; int ppid = -1;
        if (sscanf(rp + 1, " %c %d", &state, &ppid) == 2 && ppid == (int)me)
            ::kill(p, SIGKILL);
    }
    closedir(d);
}

// ---- per-component host-CPU accounting (microseconds) ----
// Threads add their own CLOCK_THREAD_CPUTIME_ID at exit; the ffmpeg/gst child
// processes are sampled once at shutdown. The AIPU's own compute is offloaded
// to silicon and intentionally NOT counted here — these are host CPU costs.
std::atomic<long long> cpu_decode_us{0};     // ffmpeg readers + decoder threads
std::atomic<long long> cpu_preproc_us{0};    // preprocess threads (resize/letterbox/quant)
std::atomic<long long> cpu_infer_us{0};      // worker thread: pack + dispatch + unpack
std::atomic<long long> cpu_draw_us{0};       // drawer thread: postproc (DFL/NMS) + box draw
std::atomic<long long> cpu_encode_us{0};     // ffmpeg writers (h264_rkmpp) + writer threads
std::atomic<long long> cpu_composite_us{0};  // disp_producer: grid assembly + HUD
std::atomic<long long> cpu_dispout_us{0};    // disp_consumer: feed the display pipe
std::atomic<long long> cpu_gst_us{0};        // gst-launch child: videoconvert + waylandsink

// CPU time consumed by the calling thread since it started, in microseconds.
long long thread_cpu_us() {
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (long long)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

// Cumulative (utime+stime) of a live pid from /proc/<pid>/stat, in microseconds.
// comm (field 2) may contain spaces/parens, so key off the last ')': the tokens
// after it are state ppid pgrp session tty tpgid flags minflt cminflt majflt
// cmajflt utime stime ... — so utime/stime are the 12th/13th tokens.
long long proc_cpu_us(pid_t pid) {
    if (pid <= 0) return 0;
    char path[64];
    std::snprintf(path, sizeof path, "/proc/%d/stat", (int)pid);
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    char buf[1024];
    char* line = fgets(buf, sizeof buf, f);
    fclose(f);
    if (!line) return 0;
    char* rp = std::strrchr(line, ')');
    if (!rp) return 0;
    char* save = nullptr;
    char* tok = strtok_r(rp + 1, " ", &save);   // state
    long vals[16]; int n = 0;
    while ((tok = strtok_r(nullptr, " ", &save)) != nullptr && n < 14)
        vals[n++] = atol(tok);                  // vals[0]=ppid ... vals[10]=utime vals[11]=stime
    if (n < 12) return 0;
    long tck = sysconf(_SC_CLK_TCK);
    if (tck <= 0) tck = 100;
    return (long long)(vals[10] + vals[11]) * 1000000 / tck;
}

// 'q'-to-quit watcher for the local fullscreen display. In a fullscreen Wayland
// session there's no terminal to receive Ctrl-C, and waylandsink ignores key
// presses, so we read the kernel evdev layer directly: every /dev/input/event*
// delivers KEY_Q regardless of which surface holds Wayland focus. We open the
// devices read-only and never EVIOCGRAB them, so Weston still sees every event
// — we're a passive second reader. On 'q' (key down) we flip g_shutdown, the
// same flag SIGINT sets, so the existing graceful teardown (MP4 trailer +
// waylandsink shutdown) runs. Needs read access to /dev/input/* (the `input`
// group on Voyager); if none are openable the thread just exits quietly and
// SIGINT/`stop_cam.sh` remain the fallback. Polls on a 200 ms timeout so it
// also exits promptly when shutdown is triggered by some other path.
void quit_key_watcher() {
    glob_t gb{};
    if (glob("/dev/input/event*", 0, nullptr, &gb) != 0) return;
    std::vector<struct pollfd> pfds;
    for (size_t i = 0; i < gb.gl_pathc; ++i) {
        int fd = open(gb.gl_pathv[i], O_RDONLY | O_NONBLOCK | O_CLOEXEC);
        if (fd >= 0) pfds.push_back({fd, POLLIN, 0});
    }
    globfree(&gb);
    if (pfds.empty()) {
        std::fprintf(stderr,
            "[quit] no readable /dev/input/event* — 'q' to quit unavailable "
            "(use Ctrl-C or stop_cam.sh)\n");
        return;
    }
    while (!g_shutdown.load(std::memory_order_relaxed)) {
        int n = poll(pfds.data(), pfds.size(), 200);
        if (n <= 0) continue;
        for (auto& pf : pfds) {
            if (!(pf.revents & POLLIN)) continue;
            struct input_event ev;
            while (read(pf.fd, &ev, sizeof ev) == (ssize_t)sizeof ev) {
                if (ev.type == EV_KEY && ev.code == KEY_Q && ev.value == 1) {
                    std::fprintf(stderr, "\n[quit] 'q' pressed — shutting down...\n");
                    g_force_quit.store(true, std::memory_order_relaxed);
                    g_shutdown.store(true, std::memory_order_relaxed);
                    for (auto& p : pfds) ::close(p.fd);
                    return;
                }
            }
        }
    }
    for (auto& p : pfds) ::close(p.fd);
}

// Per-worker bookkeeping: a dma-heap input dmabuf (mapped both as a file
// descriptor and a CPU pointer) plus heap-backed output buffers used by
// axr_run_model_instance. Allocated once per worker for the lifetime of main().
struct WorkerBufs {
    int   in_fd  = -1;
    void* in_ptr = nullptr;
    std::vector<std::vector<int8_t>> out_heap;
};

// Sample system-wide CPU% by diffing /proc/stat between two sample() calls.
// First call returns 0.0 (no baseline); subsequent calls return the percentage
// of non-idle CPU time over the interval since the previous sample.
class CpuMeter {
    uint64_t last_idle_ = 0, last_total_ = 0;
    bool     primed_ = false;
public:
    double sample() {
        FILE* fp = std::fopen("/proc/stat", "r");
        if (!fp) return 0.0;
        char tag[8];
        uint64_t user=0,nice=0,sys=0,idle=0,iow=0,irq=0,sirq=0,steal=0;
        int n = std::fscanf(fp, "%7s %lu %lu %lu %lu %lu %lu %lu %lu",
                            tag, &user, &nice, &sys, &idle, &iow, &irq, &sirq, &steal);
        std::fclose(fp);
        if (n < 5) return 0.0;
        uint64_t idle_all = idle + iow;
        uint64_t total    = user + nice + sys + idle + iow + irq + sirq + steal;
        double pct = 0.0;
        if (primed_) {
            uint64_t dt = total    - last_total_;
            uint64_t di = idle_all - last_idle_;
            if (dt > 0) pct = 100.0 * (double)(dt - di) / (double)dt;
        }
        last_idle_  = idle_all;
        last_total_ = total;
        primed_     = true;
        return pct;
    }
};

// Sample system memory usage % via /proc/meminfo. "Used" follows the same
// definition `free(1)` does: 100 * (MemTotal - MemAvailable) / MemTotal,
// which counts page cache as available rather than used.
class MemMeter {
public:
    double sample() {
        FILE* fp = std::fopen("/proc/meminfo", "r");
        if (!fp) return 0.0;
        uint64_t total = 0, avail = 0;
        char key[32]; uint64_t val;
        while (std::fscanf(fp, "%31s %lu kB\n", key, &val) == 2) {
            if      (std::strcmp(key, "MemTotal:")     == 0) total = val;
            else if (std::strcmp(key, "MemAvailable:") == 0) avail = val;
            if (total && avail) break;
        }
        std::fclose(fp);
        if (!total) return 0.0;
        if (avail > total) avail = total;
        return 100.0 * (double)(total - avail) / (double)total;
    }
};

// dma_heap_allocation_data + DMA_HEAP_IOCTL_ALLOC ABI copy (the same constants
// dma_heap.cpp uses internally). We only need them here to allocate the
// worker's per-instance batch input dmabuf.
struct dma_heap_allocation_data {
    __u64 len;
    __u32 fd;
    __u32 fd_flags;
    __u64 heap_flags;
};
constexpr unsigned long DMA_HEAP_IOCTL_ALLOC =
    _IOWR('H', 0x0, struct dma_heap_allocation_data);

void alloc_worker_bufs(std::vector<WorkerBufs>& wb,
                       size_t in_size,
                       const std::vector<size_t>& out_sizes)
{
    int heap_fd = ::open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
    if (heap_fd < 0) { std::perror("open dma_heap/system"); std::exit(1); }
    for (auto& w : wb) {
        dma_heap_allocation_data a{};
        a.len = in_size;
        a.fd_flags = O_RDWR | O_CLOEXEC;
        if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &a) < 0) {
            std::perror("worker dma alloc"); std::exit(1);
        }
        void* p = mmap(nullptr, in_size, PROT_READ | PROT_WRITE, MAP_SHARED, (int)a.fd, 0);
        if (p == MAP_FAILED) { std::perror("worker mmap"); std::exit(1); }
        w = WorkerBufs{ (int)a.fd, p, {} };
        w.out_heap.resize(out_sizes.size());
        for (size_t k = 0; k < out_sizes.size(); ++k) w.out_heap[k].resize(out_sizes[k]);
    }
    ::close(heap_fd);
}

void free_worker_bufs(std::vector<WorkerBufs>& wb, size_t in_size) {
    for (auto& w : wb) {
        if (w.in_ptr)            munmap(w.in_ptr, in_size);
        if (w.in_fd  >= 0)       ::close(w.in_fd);
    }
}

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --model PATH --inputs CSV [--record --out PREFIX] [options]\n\n"
        "Required:\n"
        "  -m, --model PATH         path to model.json or .axm\n"
        "  -i, --inputs CSV         1..40 comma-separated inputs. Each entry is one of:\n"
        "                             <path>            .mp4 / any ffmpeg-readable file\n"
        "                             usb:<N>           /dev/video<N> UVC camera (MJPEG)\n"
        "  -o, --out PREFIX         output mp4 path prefix (writes <PREFIX>_0.mp4 ...).\n"
        "                           Only used with --record; otherwise optional.\n\n"
        "Tuning:\n"
        "      --conf FLOAT         detection confidence threshold (default 0.25)\n"
        "      --iou FLOAT          NMS IoU threshold (default 0.45)\n"
        "      --fps N              per-stream target fps for ffmpeg -re -r N (default 25);\n"
        "                           for usb:<N> inputs this is the capture framerate.\n"
        "      --usb-size WxH       USB-camera capture size (default 640x480). All streams\n"
        "                           must share dimensions, so if mixing with a file, set this\n"
        "                           to match the file's resolution.\n"
        "      --unpaced            drop ffmpeg's -re flag on file inputs so frames are\n"
        "                           decoded as fast as possible (benchmark mode). Ignored\n"
        "                           for usb:<N> inputs which are always device-paced.\n"
        "      --py-dispatch        route axr_run_model_instance through a persistent\n"
        "                           tools/aipu_worker.py side-car (recovers the runtime's\n"
        "                           internal pipeline; matches axrunmodel's ~425+ fps).\n"
        "                           Output dma-bufs are mmap'd by C++ so postproc + draw +\n"
        "                           MP4 write all still work (i.e. --bench 0 is supported).\n"
        "                           Requires --workers 1.\n"
        "      --py-worker PATH     path to aipu_worker.py (default tools/aipu_worker.py)\n"
        "      --preproc N          preprocess thread count (default 4)\n\n"
        "Display / output:\n"
        "  -d, --display MODE       0=file only (default)\n"
        "                           1=local Wayland composite (waylandsink). Press 'q'\n"
        "                             on the attached keyboard to quit gracefully.\n"
        "                           2=TCP MPEG-TS H.264 composite on port 5000\n"
        "      --fullscreen         when --display 1, request a fullscreen window\n"
        "                           (waylandsink fullscreen=true). Otherwise the window\n"
        "                           is borderless via Weston's server-side decorations\n"
        "                           and the user can drag-resize it freely.\n"
        "      --boxes-only         draw per-stream colour-coded detection boxes on top\n"
        "                           of each frame. Default is clean stream pass-through\n"
        "                           with only the overall HUD on the composite.\n"
        "      --record             write one MP4 per stream (needs --out). OFF by default:\n"
        "                           the h264_rkmpp encoders are a large CPU/disk cost that a\n"
        "                           live display or benchmark run doesn't need.\n\n"
        "Diagnostic / advanced:\n"
        "  -b, --bench MODE         0=full pipeline (default)\n"
        "                           1=skip draw + write (postproc kept)\n"
        "                           2=preproc + infer only (no postproc)\n"
        "  -w, --workers N          inference instances (default 1; only meaningful\n"
        "                           with a batch=1 model, ignored for batch=4 deploys)\n\n"
        "  -h, --help               show this help and exit\n\n"
        "Examples:\n"
        "  %s -m model.json -i clip.mp4 -o /tmp/out\n"
        "  %s -m model.json -i v1.mp4,v2.mp4,v3.mp4,v4.mp4 -o /tmp/s --display 2 --fps 25\n"
        "  %s -m model.json -i usb:0 -o /tmp/cam --display 2 --usb-size 1280x720 --fps 30\n"
        "  %s -m model.json -i usb:0,usb:2,v1.mp4 -o /tmp/mix --usb-size 848x480 --display 2\n",
        prog, prog, prog, prog, prog);
}

}  // namespace


int main(int argc, char** argv) {
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa{};
    sa.sa_handler = on_sig;
    sigaction(SIGINT,  &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    // Hard-exit backstop. On a forced quit (SIGINT/SIGTERM/'q') the normal join
    // sequence drains the pipeline, but a live USB source has no EOF and an
    // occasionally-stalled encoder can back-pressure the joins. If we can't
    // exit within a few seconds, force it: _exit() closes our fds, which gives
    // the ffmpeg/gst children EOF on stdin so they still finalize the MP4
    // trailer and close the display window. Only ever armed on a forced quit,
    // so the normal EOF-completion path (which can legitimately take a moment
    // to finalize a large MP4) is never cut short.
    std::thread([]{
        while (!g_force_quit.load(std::memory_order_relaxed))
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // Scale the grace with the stream count: each stream is one more ffmpeg
        // encoder that must finalize its MP4 trailer, and at N=40 they contend
        // for the same 8 cores. This never slows the common case — on a clean
        // shutdown main exits (and the process dies) well before the grace ends.
        int n = g_stream_count.load(std::memory_order_relaxed);
        int grace_ms = 3000 + 300 * n;          // N=1 -> 3.3s, N=10 -> 6s, N=40 -> 15s
        if (grace_ms > 15000) grace_ms = 15000;
        std::this_thread::sleep_for(std::chrono::milliseconds(grace_ms));
        std::fprintf(stderr, "\n[quit] graceful drain timed out — forcing exit\n");
        std::fflush(stderr);
        kill_children();                          // no orphaned ffmpeg/gst
        _exit(0);
    }).detach();

    // ---- parse argv with getopt_long ----
    std::string model_path, in_arg, out_prefix;
    float  conf_thresh     = 0.25f;
    float  iou_thresh      = 0.45f;
    int    N               = 1;
    int    bench           = 0;
    int    preproc_threads = 4;
    int    live_display    = 0;
    int    target_fps      = 25;
    int    usb_w           = 640;
    int    usb_h           = 480;
    bool   unpaced         = false;
    bool   py_dispatch     = false;
    bool   fullscreen      = false;
    bool   boxes_only      = false;            // default: clean streams + HUD only
    bool   record          = false;            // default: do NOT write MP4s (opt in with --record)
    std::string py_worker_path = "tools/aipu_worker.py";
    int    connect_subdevs = 0;   // 0 = auto (batch*N). Set explicitly to work
                                  // around SBC drivers that refuse num_sub_devices < 4.

    enum { OPT_CONF = 1000, OPT_IOU, OPT_FPS, OPT_PREPROC, OPT_USB_SIZE,
           OPT_UNPACED, OPT_PY_DISPATCH, OPT_PY_WORKER, OPT_FULLSCREEN,
           OPT_BOXES_ONLY, OPT_CONNECT_SUBDEVS, OPT_RECORD };
    static const struct option long_opts[] = {
        {"model",    required_argument, nullptr, 'm'},
        {"inputs",   required_argument, nullptr, 'i'},
        {"out",      required_argument, nullptr, 'o'},
        {"conf",     required_argument, nullptr, OPT_CONF},
        {"iou",      required_argument, nullptr, OPT_IOU},
        {"fps",      required_argument, nullptr, OPT_FPS},
        {"preproc",  required_argument, nullptr, OPT_PREPROC},
        {"usb-size", required_argument, nullptr, OPT_USB_SIZE},
        {"unpaced",     no_argument,       nullptr, OPT_UNPACED},
        {"py-dispatch", no_argument,       nullptr, OPT_PY_DISPATCH},
        {"py-worker",   required_argument, nullptr, OPT_PY_WORKER},
        {"fullscreen",  no_argument,       nullptr, OPT_FULLSCREEN},
        {"boxes-only",  no_argument,       nullptr, OPT_BOXES_ONLY},
        {"record",      no_argument,       nullptr, OPT_RECORD},
        {"display",     required_argument, nullptr, 'd'},
        {"bench",    required_argument, nullptr, 'b'},
        {"workers",  required_argument, nullptr, 'w'},
        {"connect-subdevs", required_argument, nullptr, OPT_CONNECT_SUBDEVS},
        {"help",     no_argument,       nullptr, 'h'},
        {nullptr,    0,                 nullptr, 0  }
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "m:i:o:d:b:w:h", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'm':         model_path      = optarg;                break;
            case 'i':         in_arg          = optarg;                break;
            case 'o':         out_prefix      = optarg;                break;
            case OPT_CONF:    conf_thresh     = std::atof(optarg);     break;
            case OPT_IOU:     iou_thresh      = std::atof(optarg);     break;
            case OPT_FPS:     target_fps      = std::atoi(optarg);     break;
            case OPT_PREPROC: preproc_threads = std::atoi(optarg);     break;
            case OPT_USB_SIZE:
                if (std::sscanf(optarg, "%dx%d", &usb_w, &usb_h) != 2 || usb_w <= 0 || usb_h <= 0) {
                    std::fprintf(stderr, "ERROR: --usb-size must be WxH (e.g. 640x480)\n");
                    return 2;
                }
                break;
            case OPT_UNPACED:    unpaced        = true;                 break;
            case OPT_PY_DISPATCH: py_dispatch   = true;                 break;
            case OPT_PY_WORKER:  py_worker_path = optarg;               break;
            case OPT_FULLSCREEN: fullscreen     = true;                 break;
            case OPT_BOXES_ONLY:  boxes_only    = true;                 break;
            case OPT_RECORD:      record        = true;                 break;
            case 'd':         live_display    = std::atoi(optarg);     break;
            case 'b':         bench           = std::atoi(optarg);     break;
            case 'w':         N               = std::atoi(optarg);     break;
            case OPT_CONNECT_SUBDEVS: connect_subdevs = std::atoi(optarg); break;
            case 'h':         print_usage(argv[0]); return 0;
            default:          print_usage(argv[0]); return 2;
        }
    }
    if (optind != argc) {
        std::fprintf(stderr, "ERROR: unexpected positional argument '%s'\n\n", argv[optind]);
        print_usage(argv[0]);
        return 2;
    }
    if (model_path.empty() || in_arg.empty()) {
        std::fprintf(stderr, "ERROR: --model and --inputs are required\n\n");
        print_usage(argv[0]);
        return 2;
    }
    if (record && out_prefix.empty()) {
        std::fprintf(stderr, "ERROR: --record requires --out PREFIX\n\n");
        print_usage(argv[0]);
        return 2;
    }
    if (N < 1) N = 1;
    if (preproc_threads < 1) preproc_threads = 1;
    if (target_fps < 1) target_fps = 25;
    if (N != 1) {
        std::fprintf(stderr,
            "WARNING: --workers %d set; a batch=4 model uses all sub-devices in one\n"
            "         call and cannot benefit from multiple instances. Keep --workers 1\n"
            "         unless you have a batch=1 deploy.\n", N);
    }
    const char* model_path_c = model_path.c_str();

    // ---- TTF atlases ----
    FontAtlas* font14 = yvm::build_atlas(14.0f);
    FontAtlas* font18 = yvm::build_atlas(18.0f);
    FontAtlas* font24 = yvm::build_atlas(24.0f);  // HUD text

    // ---- parse comma-separated input paths ----
    std::vector<std::unique_ptr<Stream>> streams;
    {
        std::string buf;
        buf.reserve(in_arg.size());
        for (size_t i = 0; i <= in_arg.size(); ++i) {
            char c = (i == in_arg.size()) ? ',' : in_arg[i];
            if (c == ',') {
                if (!buf.empty()) {
                    auto s = std::make_unique<Stream>();
                    s->id      = (int)streams.size();
                    s->in_path = buf;
                    char op[512];
                    std::snprintf(op, sizeof op, "%s_%d.mp4", out_prefix.c_str(), s->id);
                    s->out_path = op;
                    streams.push_back(std::move(s));
                    buf.clear();
                }
            } else {
                buf.push_back(c);
            }
        }
    }
    if (streams.empty() || streams.size() > 40) {
        std::fprintf(stderr, "ERROR: need 1..40 input videos (got %zu)\n", streams.size());
        return 1;
    }
    g_stream_count.store((int)streams.size(), std::memory_order_relaxed);

    // ---- axruntime: 1 context/model/connection/instance (single worker; batch=4 model) ----
    axrContext* ctx = axr_create_context();
    if (!ctx) { std::fprintf(stderr, "axr_create_context failed\n"); return 1; }

    axrModel* model = axr_load_model(ctx, model_path_c);
    if (!model) {
        std::fprintf(stderr, "load_model: %s\n", axr_last_error_string(AXR_OBJECT(ctx)));
        return 1;
    }

    size_t n_in  = axr_num_model_inputs(model);
    size_t n_out = axr_num_model_outputs(model);
    std::vector<axrTensorInfo> in_infos(n_in), out_infos(n_out);
    for (size_t i = 0; i < n_in;  ++i) in_infos[i]  = axr_get_model_input(model, i);
    for (size_t i = 0; i < n_out; ++i) out_infos[i] = axr_get_model_output(model, i);
    if (n_in != 1) { std::fprintf(stderr, "expected 1 input\n"); return 1; }
    int batch = (int)in_infos[0].dims[0];

    // With --py-dispatch the AIPU is owned by the Python side-car; the C++
    // context here is used only for tensor-info introspection.
    axrConnection*               conn       = nullptr;
    axrProperties*               properties = nullptr;
    std::vector<axrModelInstance*> insts(N, nullptr);
    std::string props;
    if (!py_dispatch) {
        int connect_n = connect_subdevs > 0 ? connect_subdevs : batch * N;
        conn = axr_device_connect(ctx, nullptr, connect_n, nullptr);
        if (!conn) {
            std::fprintf(stderr, "device_connect: %s\n",
                axr_last_error_string(AXR_OBJECT(ctx)));
            return 1;
        }
        props = "input_dmabuf=1;num_sub_devices=" + std::to_string(batch)
              + ";aipu_cores="                    + std::to_string(batch)
              + ";double_buffer=1";
        properties = axr_create_properties(ctx, props.c_str());
        for (int i = 0; i < N; ++i) {
            insts[i] = axr_load_model_instance(conn, model, properties);
            if (!insts[i]) {
                std::fprintf(stderr, "load_model_instance[%d]: %s\n", i,
                    axr_last_error_string(AXR_OBJECT(ctx)));
                return 1;
            }
        }
    } else {
        if (N != 1) {
            std::fprintf(stderr,
                "ERROR: --py-dispatch only supports --workers 1\n");
            return 2;
        }
        props = "input_dmabuf=1;output_dmabuf=1;double_buffer=1 (via Python side-car)";
    }

    yvm::PreprocCtx pre_ctx = yvm::make_preproc(in_infos[0]);
    auto out_tbl = yvm::classify_outputs(out_infos);
    const size_t in_size = axr_tensor_size(&in_infos[0]);
    std::vector<size_t> out_sizes(n_out);
    for (size_t i = 0; i < n_out; ++i) out_sizes[i] = axr_tensor_size(&out_infos[i]);

    // ---- per-stream ffmpeg I/O + verify all streams share dims ----
    //
    // Each entry in --inputs is either a file path or a usb:<N> URI; the latter
    // is opened via V4L2 at --usb-size / --fps. All streams must end up at the
    // same resolution because the model preproc + composite grid assume uniform
    // per-stream dims.
    int common_sw = 0, common_sh = 0;
    double common_fps = (double)target_fps;
    for (auto& s : streams) {
        bool is_usb = s->in_path.rfind("usb:", 0) == 0;
        if (is_usb) {
            int dev_idx = std::atoi(s->in_path.c_str() + 4);
            char dev_path[32];
            std::snprintf(dev_path, sizeof dev_path, "/dev/video%d", dev_idx);
            s->sw      = usb_w;
            s->sh      = usb_h;
            s->fps_in  = (double)target_fps;
            s->nframes = 0;  // live source, unbounded
            s->reader  = yvm::ffmpeg_v4l2_reader(dev_path, usb_w, usb_h, target_fps);
        } else {
            int64_t nf = 0;
            s->reader = yvm::ffmpeg_reader(s->in_path, s->sw, s->sh, s->fps_in, nf,
                                           target_fps, unpaced);
            s->nframes = nf;
        }
        if (common_sw == 0) { common_sw = s->sw; common_sh = s->sh; }
        else if (common_sw != s->sw || common_sh != s->sh) {
            std::fprintf(stderr,
                "ERROR: stream %d is %dx%d but stream 0 is %dx%d (all inputs must match;\n"
                "       for usb:<N> sources, use --usb-size WxH to match the other inputs)\n",
                s->id, s->sw, s->sh, common_sw, common_sh);
            return 1;
        }
        // Recording is opt-in (--record). When off we skip the h264_rkmpp
        // encoder entirely — it's a big CPU cost (BGR→NV12 conversion) and disk
        // churn that a live display/benchmark run doesn't need.
        if (record) s->writer = yvm::ffmpeg_writer(s->out_path, s->sw, s->sh, common_fps);
        s->snapshot.assign((size_t)s->sw * s->sh * 3, 0);
        s->write_q = new BoundedQueue<FramePtr>(8);
    }

    std::fprintf(stderr,
        "model    : %s\n"
        "streams  : %zu  (%dx%d, %d fps each, aggregate %.0f fps)\n",
        model_path_c, streams.size(), common_sw, common_sh, target_fps,
        streams.size() * (double)target_fps);
    for (auto& s : streams)
        std::fprintf(stderr, "  [%d] %s -> %s\n", s->id, s->in_path.c_str(), s->out_path.c_str());
    std::fprintf(stderr,
        "conf=%.2f  iou=%.2f  preproc=%d  bench=%d  display=%d\nprops    : %s\n\n",
        conf_thresh, iou_thresh, preproc_threads, bench, live_display, props.c_str());

    // ---- queues + worker dmabufs ----
    BoundedQueue<FramePtr> raw_q(64);
    std::vector<std::unique_ptr<BoundedQueue<FramePtr>>> inst_q(N);
    for (int i = 0; i < N; ++i) inst_q[i] = std::make_unique<BoundedQueue<FramePtr>>(16);
    BoundedQueue<FramePtr> done_q(64);

    std::vector<WorkerBufs> wb(N);
    alloc_worker_bufs(wb, in_size, out_sizes);

    // Side-car output dma-bufs (only used in --py-dispatch mode). The runtime's
    // explore-latency sweep shows dblbuf+odmabuf=1 hits the highest system fps,
    // and Python handles the odmabuf=1 path cleanly. C++ allocates the buffers
    // (page-aligned, matching the Python DmaBufAllocator), hands the fds to the
    // worker over SCM_RIGHTS, and keeps a CPU mmap so the worker thread can
    // slice batch outputs into per-frame buffers after each ack.
    std::vector<int>    sidecar_out_fds;
    std::vector<void*>  sidecar_out_ptr;
    std::vector<size_t> sidecar_out_alloc;  // page-aligned bytes (for munmap)
    if (py_dispatch) {
        const size_t pg = (size_t)sysconf(_SC_PAGE_SIZE);
        auto page_align = [&](size_t n) { return ((n + pg - 1) / pg) * pg; };
        int heap_fd = ::open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
        if (heap_fd < 0) { std::perror("py-dispatch: open dma_heap"); return 1; }
        sidecar_out_fds.reserve(n_out);
        sidecar_out_ptr.reserve(n_out);
        sidecar_out_alloc.reserve(n_out);
        for (size_t k = 0; k < n_out; ++k) {
            size_t pa = page_align(out_sizes[k]);
            dma_heap_allocation_data a{};
            a.len      = pa;
            a.fd_flags = O_RDWR | O_CLOEXEC;
            if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &a) < 0) {
                std::perror("py-dispatch: dma alloc"); return 1;
            }
            void* p = mmap(nullptr, pa, PROT_READ, MAP_SHARED, (int)a.fd, 0);
            if (p == MAP_FAILED) { std::perror("py-dispatch: mmap out"); return 1; }
            sidecar_out_fds.push_back((int)a.fd);
            sidecar_out_ptr.push_back(p);
            sidecar_out_alloc.push_back(pa);
        }
        ::close(heap_fd);
    }

    // Start the Python side-car (after dma-bufs are allocated, so we can pass
    // the fds over SCM_RIGHTS). It loads the model + claims the AIPU.
    yvm::PyAipuClient py_cli;
    if (py_dispatch) {
        std::fprintf(stderr, "[py-dispatch] starting %s ...\n", py_worker_path.c_str());
        if (!py_cli.start(py_worker_path, model_path, batch, batch,
                          (int)n_out, /*output_dmabuf=*/true,
                          wb[0].in_fd, sidecar_out_fds)) {
            std::fprintf(stderr, "[py-dispatch] worker setup failed\n");
            return 1;
        }
        std::fprintf(stderr, "[py-dispatch] worker ready (output_dmabuf=1)\n");
    }

    std::atomic<int64_t> infer_ns_total{0};
    std::atomic<int>     frames_inferred{0};

    // ---- pipeline latency instrumentation ----
    //
    // We attach a steady_clock timestamp to each Frame in the decoder thread
    // (Frame::t_arr) and compute two derived latencies later in the pipeline:
    //
    //   v4l2->drawn:   from t_arr to drawer finishing postproc + box draw
    //                  (covers preproc, AIPU dispatch, postproc, box draw)
    //   v4l2->gst:     from t_arr to the moment the display consumer writes
    //                  the composite that contains this frame into gst-launch
    //                  stdin (covers everything above + composite sampling
    //                  jitter + LeakyOne buffering)
    //
    // The display consumer cannot easily look up which Frame is in the latest
    // composite, so the drawer publishes its most recent stream-0 t_arr into
    // `latest_drawn_arr_steady_ns` (nanoseconds since steady_clock epoch) and
    // the consumer reads that to compute the v4l2->gst latency.
    std::atomic<int64_t> sum_ns_arr_to_drawn{0};
    std::atomic<int64_t> n_arr_to_drawn{0};
    std::atomic<int64_t> max_ns_arr_to_drawn{0};
    std::atomic<int64_t> sum_ns_arr_to_disp{0};
    std::atomic<int64_t> n_arr_to_disp{0};
    std::atomic<int64_t> max_ns_arr_to_disp{0};
    // HUD mirror — reset by the 1-Hz HUD tick, independent of the stats logger.
    std::atomic<int64_t> hud_sum_ns_arr_to_disp{0};
    std::atomic<int64_t> hud_n_arr_to_disp{0};
    std::atomic<int64_t> latest_drawn_arr_steady_ns{0};
    auto update_max_atomic = [](std::atomic<int64_t>& m, int64_t v) {
        int64_t prev = m.load(std::memory_order_relaxed);
        while (v > prev && !m.compare_exchange_weak(prev, v, std::memory_order_relaxed)) {}
    };

    // ---- event-driven display wake-up ----
    //
    // The drawer increments `disp_dirty` and signals `disp_cv` after each
    // fresh snapshot. The display producer waits on the cv instead of polling
    // at a fixed 30 Hz, cutting up to ~33 ms of pipeline latency in the
    // single-stream case. A `max_idle` timeout still fires periodically so
    // the HUD's once-per-second numbers keep refreshing when no streams
    // produce frames.
    std::mutex             disp_cv_mu;
    std::condition_variable disp_cv;
    std::atomic<int>       disp_dirty{0};
    auto t0 = clk::now();

    // ---- stats logger (every 2 s) + shutdown watcher ----
    std::thread stats_t([&]() {
        auto t_prev = clk::now();
        int last_inf = 0;
        std::vector<int> last_drawn(streams.size(), 0);
        while (!g_shutdown.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            auto now = clk::now();
            double dt = std::chrono::duration<double>(now - t_prev).count();
            t_prev = now;
            int inf_now = frames_inferred.load();
            double infer_fps = (inf_now - last_inf) / dt;
            last_inf = inf_now;
            std::string per_stream;
            double agg = 0;
            for (size_t i = 0; i < streams.size(); ++i) {
                int dn = streams[i]->drawn.load();
                double sf = (dn - last_drawn[i]) / dt;
                last_drawn[i] = dn; agg += sf;
                char buf[32]; std::snprintf(buf, sizeof buf, " s%zu=%.1f", i, sf);
                per_stream += buf;
            }
            std::fprintf(stderr, "[stats] infer=%.1f fps  agg-drawn=%.1f fps  per-stream:%s\n",
                         infer_fps, agg, per_stream.c_str());

            // Latency summary (resets accumulators each tick).
            int64_t n_d = n_arr_to_drawn.exchange(0, std::memory_order_relaxed);
            int64_t s_d = sum_ns_arr_to_drawn.exchange(0, std::memory_order_relaxed);
            int64_t m_d = max_ns_arr_to_drawn.exchange(0, std::memory_order_relaxed);
            int64_t n_p = n_arr_to_disp.exchange(0, std::memory_order_relaxed);
            int64_t s_p = sum_ns_arr_to_disp.exchange(0, std::memory_order_relaxed);
            int64_t m_p = max_ns_arr_to_disp.exchange(0, std::memory_order_relaxed);
            if (n_d > 0) {
                std::fprintf(stderr,
                    "[lat ] v4l2->drawn  mean=%.2f ms  max=%.2f ms"
                    "   v4l2->gst-stdin  mean=%.2f ms  max=%.2f ms"
                    "  (excludes USB capture + waylandsink/DRM/panel)\n",
                    s_d / 1e6 / (double)n_d, m_d / 1e6,
                    n_p > 0 ? s_p / 1e6 / (double)n_p : 0.0,
                    m_p / 1e6);
            }
        }
    });
    std::thread sig_watcher([&]() {
        while (!g_shutdown.load(std::memory_order_relaxed))
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        // Sample child-process CPU for the end-of-run breakdown while they are
        // still alive (the reaping path uses plain waitpid, no rusage). Do this
        // before any SIGTERM below.
        for (auto& sp : streams) {
            cpu_decode_us.fetch_add(proc_cpu_us(sp->reader.pid));
            cpu_encode_us.fetch_add(proc_cpu_us(sp->writer.pid));
        }
        { long long c = proc_cpu_us(g_disp_pid.load(std::memory_order_relaxed));
          if (c > 0) cpu_gst_us.store(c); }
        // A live USB source never hits EOF, so wake the v4l2 readers to unblock
        // the decoders' read_full().
        for (auto& sp : streams)
            if (sp->reader.pid > 0) ::kill(sp->reader.pid, SIGTERM);
        // Drop in-flight work: closing every queue wakes any thread blocked in
        // push()/pop() so the join sequence can't wedge on a full downstream
        // queue (e.g. a momentarily-stalled writer back-pressuring the drawer →
        // workers → preprocs). close() is idempotent, so this is harmless on the
        // normal EOF path where these are already closed/drained.
        raw_q.close();
        for (auto& q : inst_q) q->close();
        done_q.close();
        for (auto& sp : streams)
            if (sp->write_q) sp->write_q->close();
        // On a *forced* quit, closing the queues can't interrupt a thread
        // already blocked in a write() to a child's pipe (a stalled h264_rkmpp
        // encoder, or a frozen waylandsink). SIGTERM the children: ffmpeg
        // finalizes its MP4 trailer cleanly on SIGTERM, and either way the pipe
        // breaks so the writer / display threads get EPIPE and the joins
        // complete promptly. The normal EOF path finalizes via stdin-close
        // instead (all frames flushed), so it's left untouched here.
        if (g_force_quit.load(std::memory_order_relaxed)) {
            for (auto& sp : streams)
                if (sp->writer.pid > 0) ::kill(sp->writer.pid, SIGTERM);
            pid_t dp = g_disp_pid.load(std::memory_order_relaxed);
            if (dp > 0) ::kill(dp, SIGTERM);
        }
    });

    // ---- decoder threads (one per stream) ----
    std::vector<std::thread> decoders;
    for (auto& sp : streams) {
        Stream* s = sp.get();
        decoders.emplace_back([&, s]() {
            const size_t frame_bytes = (size_t)s->sw * s->sh * 3;
            int idx = 0;
            while (!g_shutdown.load(std::memory_order_relaxed)) {
                auto f = std::make_unique<Frame>();
                f->bgr.resize(frame_bytes);
                f->sw = s->sw; f->sh = s->sh;
                f->stream_id = s->id;
                if (!yvm::read_full(s->reader.fd, f->bgr.data(), frame_bytes)) break;
                f->t_arr = std::chrono::steady_clock::now();
                f->idx = idx++;
                s->decoded.fetch_add(1, std::memory_order_relaxed);
                raw_q.push(std::move(f));
            }
            cpu_decode_us.fetch_add(thread_cpu_us());
        });
    }

    // ---- preproc threads (shared raw_q) ----
    std::vector<std::thread> preprocs;
    std::atomic<int> preproc_done{0};
    for (int t = 0; t < preproc_threads; ++t) {
        preprocs.emplace_back([&]() {
            FramePtr f;
            while (raw_q.pop(f)) {
                f->outputs.resize(n_out);
                f->in_args.assign(1, {});
                f->out_args.assign(n_out, {});
                f->input.assign(in_size / batch, 0);
                yvm::preprocess(f->bgr.data(), f->sw, f->sh,
                                reinterpret_cast<int8_t*>(f->input.data()),
                                pre_ctx, f->lscale, f->padx, f->pady);
                f->in_args[0].ptr  = f->input.data();
                f->in_args[0].size = f->input.size();
                for (size_t k = 0; k < n_out; ++k) {
                    f->outputs[k].assign(out_sizes[k], 0);
                    f->out_args[k].ptr  = f->outputs[k].data();
                    f->out_args[k].size = f->outputs[k].size();
                }
                inst_q[0]->push(std::move(f));
            }
            if (preproc_done.fetch_add(1) + 1 == preproc_threads) {
                for (auto& q : inst_q) q->close();
            }
            cpu_preproc_us.fetch_add(thread_cpu_us());
        });
    }

    // ---- worker thread(s): pack 4 frames per inference call ----
    std::vector<std::thread> workers;
    for (int i = 0; i < N; ++i) {
        workers.emplace_back([&, i]() {
            const int    B = batch;
            const size_t in_slot = in_size / B;
            std::vector<size_t> out_slot(n_out);
            for (size_t k = 0; k < n_out; ++k) out_slot[k] = out_sizes[k] / B;

            std::vector<axrArgument> in_args(1), out_args(n_out);
            in_args[0].ptr    = nullptr;
            in_args[0].fd     = wb[i].in_fd;
            in_args[0].offset = 0;
            in_args[0].size   = in_size;
            for (size_t k = 0; k < n_out; ++k) {
                out_args[k].ptr    = wb[i].out_heap[k].data();
                out_args[k].size   = wb[i].out_heap[k].size();
                out_args[k].fd     = 0;
                out_args[k].offset = 0;
            }

            std::vector<FramePtr> batch_frames;
            batch_frames.reserve(B);

            auto run_batch = [&]() {
                if (batch_frames.empty()) return;
                InputBufferPool::sync_start_write(wb[i].in_fd);
                for (int b = 0; b < (int)batch_frames.size(); ++b) {
                    std::memcpy(static_cast<uint8_t*>(wb[i].in_ptr) + b * in_slot,
                                batch_frames[b]->input.data(), in_slot);
                }
                // Replicate slot 0 into any unused tail slots in this final partial batch.
                for (int b = (int)batch_frames.size(); b < B; ++b) {
                    std::memcpy(static_cast<uint8_t*>(wb[i].in_ptr) + b * in_slot,
                                wb[i].in_ptr, in_slot);
                }
                InputBufferPool::sync_end_write(wb[i].in_fd);

                auto t_a = clk::now();
                bool ok;
                if (py_dispatch) {
                    ok = py_cli.run_one();
                } else {
                    axrResult r = axr_run_model_instance(insts[i],
                        in_args.data(), in_args.size(),
                        out_args.data(), out_args.size());
                    ok = (r == AXR_SUCCESS);
                    if (!ok) {
                        std::fprintf(stderr, "[worker %d] run failed: code=%d (%s) msg=%s\n",
                            i, (int)r, axr_error_string(r),
                            axr_last_error_string(AXR_OBJECT(ctx)));
                    }
                }
                auto t_b = clk::now();
                if (!ok) {
                    if (py_dispatch)
                        std::fprintf(stderr, "[worker %d] py-dispatch run failed\n", i);
                    return;
                }
                infer_ns_total.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t_b - t_a).count(),
                    std::memory_order_relaxed);

                // Outputs land in either:
                //   - wb[i].out_heap[k]      (public C API path, runtime memcpys here)
                //   - sidecar_out_ptr[k]     (py-dispatch path, AIPU DMAs here; invalidate caches)
                if (py_dispatch) {
                    for (size_t k = 0; k < n_out; ++k)
                        InputBufferPool::sync_start_read(sidecar_out_fds[k]);
                }
                for (int b = 0; b < (int)batch_frames.size(); ++b) {
                    auto& f = batch_frames[b];
                    f->outputs.resize(n_out);
                    for (size_t k = 0; k < n_out; ++k) {
                        f->outputs[k].assign(out_slot[k], 0);
                        const uint8_t* src = py_dispatch
                            ? static_cast<const uint8_t*>(sidecar_out_ptr[k])
                            : reinterpret_cast<const uint8_t*>(wb[i].out_heap[k].data());
                        std::memcpy(f->outputs[k].data(), src + b * out_slot[k], out_slot[k]);
                    }
                    frames_inferred.fetch_add(1, std::memory_order_relaxed);
                    done_q.push(std::move(f));
                }
                if (py_dispatch) {
                    for (size_t k = 0; k < n_out; ++k)
                        InputBufferPool::sync_end_read(sidecar_out_fds[k]);
                }
                batch_frames.clear();
            };

            FramePtr f;
            while (inst_q[i]->pop(f)) {
                batch_frames.push_back(std::move(f));
                if ((int)batch_frames.size() == B) run_batch();
            }
            run_batch();  // flush any partial-batch leftover at EOF
            cpu_infer_us.fetch_add(thread_cpu_us());
        });
    }

    // ---- drawer thread: per-stream reorder + draw + push to per-stream write_q + snapshot ----
    std::thread drawer([&]() {
        FramePtr f;
        std::vector<Detection> dets;
        while (done_q.pop(f)) {
            int sid = f->stream_id;
            Stream* s = streams[sid].get();
            s->pending.emplace(f->idx, std::move(f));
            while (true) {
                auto it = s->pending.find(s->next_idx);
                if (it == s->pending.end()) break;
                FramePtr cur = std::move(it->second);
                s->pending.erase(it);
                ++s->next_idx;

                // Default is clean stream pass-through; only the global HUD on
                // the composite remains. Opt-in to per-stream colour-coded boxes
                // with --boxes-only.
                if (boxes_only && bench < 2) {
                    std::vector<const int8_t*> ptrs(cur->outputs.size());
                    for (size_t k = 0; k < cur->outputs.size(); ++k) ptrs[k] = cur->outputs[k].data();
                    dets.clear();
                    yvm::decode_dfl_sigmoid_filter(ptrs, out_infos, out_tbl,
                                                   conf_thresh, cur->lscale, cur->padx, cur->pady,
                                                   cur->sw, cur->sh, dets);
                    auto kept = yvm::nms(std::move(dets), iou_thresh);

                    if (bench == 0) {
                        // Boxes only — class label and confidence used to be drawn
                        // as a small chip at the top-left corner of each box, but
                        // those text bubbles clutter the view at small cell sizes.
                        // Box colour already encodes the class (see class_color).
                        Image im{cur->bgr.data(), cur->sw, cur->sh};
                        for (const auto& d : kept) {
                            uint8_t bc, gc, rc;
                            yvm::class_color(d.cls, bc, gc, rc);
                            int x1 = (int)d.x1, y1 = (int)d.y1,
                                x2 = (int)d.x2, y2 = (int)d.y2;
                            yvm::draw_rect(im, x1, y1, x2, y2, bc, gc, rc, 2);
                        }
                    }
                }

                s->drawn.fetch_add(1, std::memory_order_relaxed);

                // Latency accounting: from V4L2 arrival to here (drawer done).
                {
                    auto now = std::chrono::steady_clock::now();
                    int64_t lat_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         now - cur->t_arr).count();
                    sum_ns_arr_to_drawn.fetch_add(lat_ns, std::memory_order_relaxed);
                    n_arr_to_drawn.fetch_add(1, std::memory_order_relaxed);
                    update_max_atomic(max_ns_arr_to_drawn, lat_ns);
                    // Publish stream-0's arrival timestamp for the display
                    // consumer to read when it next writes a composite.
                    if (cur->stream_id == 0) {
                        latest_drawn_arr_steady_ns.store(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                cur->t_arr.time_since_epoch()).count(),
                            std::memory_order_relaxed);
                    }
                }

                if (live_display) {
                    {
                        std::lock_guard<std::mutex> lk(s->snap_mu);
                        std::memcpy(s->snapshot.data(), cur->bgr.data(),
                                    std::min(s->snapshot.size(), cur->bgr.size()));
                    }
                    // Wake the display producer event-driven (was 30 Hz poll).
                    {
                        std::lock_guard<std::mutex> lk(disp_cv_mu);
                        disp_dirty.fetch_add(1, std::memory_order_relaxed);
                    }
                    disp_cv.notify_one();
                }
                if (record) s->write_q->push(std::move(cur));
                // else: cur is dropped here (no MP4 writer to consume it).
            }
        }
        for (auto& sp : streams) sp->write_q->close();
        cpu_draw_us.fetch_add(thread_cpu_us());
    });

    // ---- per-stream writer threads (only when --record) ----
    if (record) {
        for (auto& sp : streams) {
            Stream* s = sp.get();
            s->writer_thread = std::thread([&, s]() {
                FramePtr f;
                while (s->write_q->pop(f)) {
                    if (bench != 0) continue;
                    if (!yvm::write_full(s->writer.fd, f->bgr.data(), f->bgr.size())) break;
                    s->written.fetch_add(1, std::memory_order_relaxed);
                }
                cpu_encode_us.fetch_add(thread_cpu_us());
            });
        }
    }

    // ---- composite display: producer @ 30 Hz, consumer with auto-respawn ----
    //
    // Grid is chosen automatically to fit N streams:
    //
    //     cols = ceil(sqrt(N))      rows = ceil(N / cols)      cells = cols * rows
    //
    // Examples (matches the layout voyager-sdk's display.App uses for an OpenGL window):
    //   N=1  → 1x1  (single full-size view)
    //   N=2  → 2x1
    //   N=3  → 2x2  (one black cell)
    //   N=4  → 2x2
    //   N=5,6 → 3x2
    //   N=7,8,9 → 3x3
    //   N=10..12 → 4x3
    //   N=13..16 → 4x4
    //
    // Each cell is the source frame downscaled by integer factors (scale_x=cols, scale_y=rows),
    // so for N=1 the cell IS the source frame and no resize happens.
    LeakyOne<std::vector<uint8_t>> disp_slot;
    std::thread disp_producer, disp_consumer, quit_watcher;
    std::atomic<bool> disp_stop{false};

    // Local fullscreen has no terminal for Ctrl-C, so watch evdev for 'q'.
    if (live_display == 1)
        quit_watcher = std::thread(quit_key_watcher);
    const int N_streams = (int)streams.size();
    const int GRID_COLS = (int)std::ceil(std::sqrt((double)N_streams));
    const int GRID_ROWS = (int)std::ceil((double)N_streams / GRID_COLS);
    int cell_w = 0, cell_h = 0, comp_w = 0, comp_h = 0, scale_x = 0, scale_y = 0;
    if (live_display) {
        cell_w  = common_sw / GRID_COLS;
        cell_h  = common_sh / GRID_ROWS;
        scale_x = common_sw / cell_w;
        scale_y = common_sh / cell_h;
        comp_w  = GRID_COLS * cell_w;
        comp_h  = GRID_ROWS * cell_h;

        disp_producer = std::thread([&]() {
            std::vector<uint8_t> composite((size_t)comp_w * comp_h * 3, 0);
            using clk2 = std::chrono::steady_clock;
            // Event-driven push, rate-capped to ~60 Hz so multi-stream doesn't
            // hammer the sink. `max_idle` ensures the HUD keeps refreshing even
            // if no streams produce frames (e.g., everything stalled).
            const auto min_gap  = std::chrono::milliseconds(16);
            const auto max_idle = std::chrono::milliseconds(200);
            auto last_push = clk2::now();
            std::vector<std::vector<uint8_t>> snaps(streams.size());
            for (auto& sn : snaps) sn.assign((size_t)common_sw * common_sh * 3, 0);

            // Overall HUD state: sample once per second so the readout is steady
            // (we still redraw the last sampled values onto every composite frame).
            CpuMeter cpu;
            MemMeter mem;
            auto hud_t_prev = clk2::now();
            int  hud_last_inf = frames_inferred.load(std::memory_order_relaxed);
            std::vector<int> hud_last_drawn(streams.size(), 0);
            for (size_t i = 0; i < streams.size(); ++i)
                hud_last_drawn[i] = streams[i]->drawn.load(std::memory_order_relaxed);
            double hud_e2e_fps = 0.0, hud_inf_fps = 0.0,
                   hud_cpu_pct = 0.0, hud_mem_pct = 0.0,
                   hud_lat_ms = 0.0;

            while (!disp_stop.load(std::memory_order_relaxed)) {
                // Wait for the drawer to signal a new snapshot (or timeout for
                // HUD refresh / stop check). Then enforce the 60 Hz rate cap.
                {
                    std::unique_lock<std::mutex> lk(disp_cv_mu);
                    disp_cv.wait_for(lk, max_idle, [&]{
                        return disp_dirty.load(std::memory_order_relaxed) != 0
                            || disp_stop.load(std::memory_order_relaxed);
                    });
                    disp_dirty.store(0, std::memory_order_relaxed);
                }
                if (disp_stop.load(std::memory_order_relaxed)) break;
                auto rate_target = last_push + min_gap;
                if (clk2::now() < rate_target)
                    std::this_thread::sleep_until(rate_target);
                last_push = clk2::now();

                for (size_t i = 0; i < streams.size(); ++i) {
                    Stream* s = streams[i].get();
                    {
                        std::lock_guard<std::mutex> lk(s->snap_mu);
                        if (s->snapshot.size() == snaps[i].size())
                            std::memcpy(snaps[i].data(), s->snapshot.data(), snaps[i].size());
                    }
                    int gx = (int)i % GRID_COLS, gy = (int)i / GRID_COLS;
                    if (gy >= GRID_ROWS) continue;
                    int dst_x0 = gx * cell_w, dst_y0 = gy * cell_h;
                    for (int y = 0; y < cell_h; ++y) {
                        const uint8_t* src = snaps[i].data() + (size_t)(y * scale_y) * s->sw * 3;
                        uint8_t* dst = composite.data()
                                     + ((size_t)(dst_y0 + y) * comp_w + dst_x0) * 3;
                        for (int x = 0; x < cell_w; ++x) {
                            const uint8_t* p = src + (x * scale_x) * 3;
                            dst[0] = p[0]; dst[1] = p[1]; dst[2] = p[2];
                            dst += 3;
                        }
                    }
                }

                // Refresh HUD numbers once per second.
                auto now = clk2::now();
                double dt = std::chrono::duration<double>(now - hud_t_prev).count();
                if (dt >= 1.0) {
                    int inf_now = frames_inferred.load(std::memory_order_relaxed);
                    hud_inf_fps = (inf_now - hud_last_inf) / dt;
                    hud_last_inf = inf_now;
                    double agg = 0.0;
                    for (size_t i = 0; i < streams.size(); ++i) {
                        int dn = streams[i]->drawn.load(std::memory_order_relaxed);
                        agg += (dn - hud_last_drawn[i]) / dt;
                        hud_last_drawn[i] = dn;
                    }
                    hud_e2e_fps = agg;
                    hud_cpu_pct = cpu.sample();
                    hud_mem_pct = mem.sample();
                    int64_t hn = hud_n_arr_to_disp.exchange(0, std::memory_order_relaxed);
                    int64_t hs = hud_sum_ns_arr_to_disp.exchange(0, std::memory_order_relaxed);
                    if (hn > 0) hud_lat_ms = (hs / 1e6) / (double)hn;
                    hud_t_prev  = now;
                }

                // Two-line HUD panel, right-aligned values for a clean stat-block look:
                //
                //     ┌──────────────────────────┐
                //     │  E2E    286 fps          │
                //     │  Infer  286 fps          │
                //     │  CPU       78 %          │
                //     │  MEM       62 %          │
                //     └──────────────────────────┘
                FontAtlas* fhud = font24 ? font24 : font18;
                struct Row { char label[16]; char value[24]; };
                Row rows[5] = {};
                std::snprintf(rows[0].label, sizeof rows[0].label, "E2E");
                std::snprintf(rows[0].value, sizeof rows[0].value, "%.0f fps", hud_e2e_fps);
                std::snprintf(rows[1].label, sizeof rows[1].label, "Infer");
                std::snprintf(rows[1].value, sizeof rows[1].value, "%.0f fps", hud_inf_fps);
                std::snprintf(rows[2].label, sizeof rows[2].label, "Lat");
                std::snprintf(rows[2].value, sizeof rows[2].value, "%.0f ms",  hud_lat_ms);
                std::snprintf(rows[3].label, sizeof rows[3].label, "CPU");
                std::snprintf(rows[3].value, sizeof rows[3].value, "%.0f %%",  hud_cpu_pct);
                std::snprintf(rows[4].label, sizeof rows[4].label, "MEM");
                std::snprintf(rows[4].value, sizeof rows[4].value, "%.0f %%",  hud_mem_pct);

                const int rows_n   = 5;
                const int pad_x    = 18;
                const int pad_y    = 14;
                const int gap_lbl  = 28;             // gap between label and value column
                const int line_h   = fhud ? (fhud->line_height + 6) : 28;
                int label_w = 0, value_w = 0;
                for (int r = 0; r < rows_n; ++r) {
                    int lw = fhud ? yvm::text_width(*fhud, rows[r].label) : (int)std::strlen(rows[r].label) * 12;
                    int vw = fhud ? yvm::text_width(*fhud, rows[r].value) : (int)std::strlen(rows[r].value) * 12;
                    if (lw > label_w) label_w = lw;
                    if (vw > value_w) value_w = vw;
                }
                int panel_w = pad_x + label_w + gap_lbl + value_w + pad_x;
                int panel_h = pad_y + rows_n * line_h + pad_y - 6;
                int x0 = 12, y0 = 12;
                yvm::fill_rect_alpha(composite.data(), comp_w, comp_h,
                                     x0, y0, x0 + panel_w, y0 + panel_h,
                                     0, 0, 0, 200);
                if (fhud) {
                    int label_x = x0 + pad_x;
                    int value_x = x0 + pad_x + label_w + gap_lbl;
                    for (int r = 0; r < rows_n; ++r) {
                        int y = y0 + pad_y + r * line_h;
                        // Subtle grey label, bright white value — typical stat-block style.
                        yvm::draw_text_shadow(composite.data(), comp_w, comp_h,
                                              label_x, y, rows[r].label,
                                              180, 180, 180, *fhud);
                        // Right-align the value column.
                        int vw = yvm::text_width(*fhud, rows[r].value);
                        yvm::draw_text_shadow(composite.data(), comp_w, comp_h,
                                              value_x + value_w - vw, y, rows[r].value,
                                              255, 255, 255, *fhud);
                    }
                }

                disp_slot.push(std::vector<uint8_t>(composite));
            }
            cpu_composite_us.fetch_add(thread_cpu_us());
        });

        disp_consumer = std::thread([&]() {
            Subprocess disp{};
            auto open_disp = [&]() {
                if (live_display == 1)
                    disp = yvm::gst_local_display(comp_w, comp_h, 30.0,
                                                  "yolo11n multi", fullscreen);
                else
                    disp = yvm::ffmpeg_tcp_streamer(comp_w, comp_h, 30.0, 5000);
                g_disp_pid.store(disp.pid, std::memory_order_relaxed);
                std::fprintf(stderr,
                    "[display] composite pid=%d  %dx%d  %dx%d grid  cell %dx%d  (scale 1/%dx1/%d)\n",
                    (int)disp.pid, comp_w, comp_h, GRID_COLS, GRID_ROWS,
                    cell_w, cell_h, scale_x, scale_y);
            };
            open_disp();
            std::vector<uint8_t> frame;
            while (!disp_stop.load(std::memory_order_relaxed) && disp_slot.pop(frame)) {
                // On shutdown stop feeding the sink and don't respawn it — let
                // sig_watcher's SIGTERM (or close_sub below) tear the child down.
                if (g_shutdown.load(std::memory_order_relaxed)) break;
                if (disp.fd < 0) open_disp();
                // Sample latest drawn-frame arrival just before we hand the
                // composite to gst-launch. This is the "v4l2 -> gst stdin"
                // latency; it does NOT include GStreamer plugin chain, DRM
                // commit, or monitor scan-out time (~one frame at panel rate).
                int64_t arr_ns = latest_drawn_arr_steady_ns.load(std::memory_order_relaxed);
                if (!yvm::write_full(disp.fd, frame.data(), frame.size())) {
                    std::fprintf(stderr, "[display] viewer disconnected; respawning...\n");
                    yvm::close_sub(disp);
                } else if (arr_ns > 0) {
                    int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
                    int64_t lat_ns = now_ns - arr_ns;
                    if (lat_ns >= 0 && lat_ns < (int64_t)5e9) {  // sanity clamp
                        sum_ns_arr_to_disp.fetch_add(lat_ns, std::memory_order_relaxed);
                        n_arr_to_disp.fetch_add(1, std::memory_order_relaxed);
                        update_max_atomic(max_ns_arr_to_disp, lat_ns);
                        hud_sum_ns_arr_to_disp.fetch_add(lat_ns, std::memory_order_relaxed);
                        hud_n_arr_to_disp.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
            // Sample the gst child's CPU while it's (usually) still alive, then
            // tear it down. sig_watcher may already have sampled+killed it on a
            // forced quit, so only record a non-zero reading (don't overwrite).
            { long long c = proc_cpu_us(disp.pid); if (c > 0) cpu_gst_us.store(c); }
            cpu_dispout_us.fetch_add(thread_cpu_us());
            yvm::close_sub(disp);
        });
    }

    // ---- join + graceful shutdown ----
    for (auto& t : decoders) t.join();
    raw_q.close();
    for (auto& t : preprocs) t.join();
    for (auto& t : workers)  t.join();
    done_q.close();
    drawer.join();
    for (auto& sp : streams) if (sp->writer_thread.joinable()) sp->writer_thread.join();
    g_shutdown.store(true);
    if (stats_t.joinable())     stats_t.join();
    if (sig_watcher.joinable()) sig_watcher.join();

    auto t1 = clk::now();
    double total_s = std::chrono::duration<double>(t1 - t0).count();
    int inf_n      = frames_inferred.load();
    int64_t inf_ns = infer_ns_total.load();

    std::fprintf(stderr, "\n=== done in %.2f s ===\n", total_s);
    for (auto& sp : streams) {
        Stream* s = sp.get();
        int w = s->written.load();
        std::fprintf(stderr, "  stream %d:  decoded %d  drawn %d  written %d  (%.1f fps)\n",
                     s->id, s->decoded.load(), s->drawn.load(), w, w / total_s);
    }
    std::fprintf(stderr,
        "aggregate: %d frames inferred,  effective infer rate %.1f fps  (single-call lat %.3f ms),\n"
        "           wall-clock infer throughput %.1f fps\n",
        inf_n,
        (inf_ns && inf_n) ? 1e9 * inf_n / (double)inf_ns : 0.0,
        (inf_ns && inf_n) ? inf_ns / 1e6 / inf_n : 0.0,
        inf_n / total_s);

    // Parallel writer-stdin close so every ffmpeg child finalizes its mp4 trailer in parallel.
    for (auto& sp : streams) {
        if (sp->reader.fd >= 0) { ::close(sp->reader.fd); sp->reader.fd = -1; }
        if (sp->writer.fd >= 0) { ::close(sp->writer.fd); sp->writer.fd = -1; }
    }
    for (auto& sp : streams) {
        for (Subprocess* sub : { &sp->reader, &sp->writer }) {
            if (sub->pid <= 0) continue;
            int st = 0; bool done = false;
            for (int i = 0; i < 150 && !done; ++i) {            // up to 15 s clean
                pid_t r = waitpid(sub->pid, &st, WNOHANG);
                if (r == sub->pid) { done = true; break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (!done) {
                ::kill(sub->pid, SIGTERM);
                for (int i = 0; i < 30 && !done; ++i) {
                    pid_t r = waitpid(sub->pid, &st, WNOHANG);
                    if (r == sub->pid) { done = true; break; }
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
            if (!done) { ::kill(sub->pid, SIGKILL); waitpid(sub->pid, &st, 0); }
            sub->pid = -1;
        }
    }
    if (live_display) {
        disp_stop.store(true);
        disp_cv.notify_all();    // wake the event-driven producer
        disp_slot.close();
        if (disp_producer.joinable()) disp_producer.join();
        if (disp_consumer.joinable()) disp_consumer.join();
    }
    // g_shutdown is already set above, so the evdev watcher exits its next poll.
    if (quit_watcher.joinable()) quit_watcher.join();

    // ---- per-component host-CPU breakdown ----
    // Printed here, after every worker/display thread has joined, so each has
    // recorded its CLOCK_THREAD_CPUTIME_ID. cpu-seconds are host CPU only; the
    // AIPU's own matmul/conv work runs on silicon and is NOT counted, so
    // "inference" is the host cost of feeding/draining the AIPU (preproc +
    // dispatch + memcpys + postproc), not the model maths itself.
    {
        long ncpu = sysconf(_SC_NPROCESSORS_ONLN); if (ncpu < 1) ncpu = 1;
        double W = total_s > 1e-6 ? total_s : 1e-6;
        double dec_s = cpu_decode_us.load()    / 1e6;
        double pre_s = cpu_preproc_us.load()   / 1e6;
        double inf_s = cpu_infer_us.load()     / 1e6;
        double drw_s = cpu_draw_us.load()      / 1e6;
        double enc_s = cpu_encode_us.load()    / 1e6;
        double cmp_s = cpu_composite_us.load() / 1e6;
        double out_s = cpu_dispout_us.load()   / 1e6;
        double gst_s = cpu_gst_us.load()       / 1e6;
        double infer_side   = pre_s + inf_s + drw_s;     // host side of the model pipeline
        double display_side = cmp_s + out_s + gst_s;     // compositing + sink
        std::fprintf(stderr,
            "\n=== host CPU breakdown over %.1f s wall, %ld cores (AIPU compute offloaded, not counted) ===\n"
            "  component                       cpu-s   %%1core  cores\n"
            "  decode   (ffmpeg readers)      %7.1f  %5.0f%%  %5.2f\n"
            "  preprocess                     %7.1f  %5.0f%%  %5.2f\n"
            "  inference dispatch (worker)    %7.1f  %5.0f%%  %5.2f\n"
            "  postproc + box draw            %7.1f  %5.0f%%  %5.2f\n"
            "  composite (grid + HUD)         %7.1f  %5.0f%%  %5.2f\n"
            "  display feed (disp consumer)   %7.1f  %5.0f%%  %5.2f\n"
            "  display sink (gst convert)     %7.1f  %5.0f%%  %5.2f\n"
            "  encode/record (ffmpeg)         %7.1f  %5.0f%%  %5.2f\n"
            "  -----------------------------------------------------\n"
            "  INFERENCE (preproc+dispatch+postproc)  %7.1f cpu-s  %5.2f cores  (%.0f%% of infer+display)\n"
            "  DISPLAY   (composite+feed+sink)        %7.1f cpu-s  %5.2f cores  (%.0f%% of infer+display)\n"
            "  decode %.1f cpu-s,  encode/record %.1f cpu-s  (input/output, not infer or display)\n"
            "  inference : display  =  %.2f : 1\n",
            total_s, ncpu,
            dec_s, 100*dec_s/W, dec_s/W,
            pre_s, 100*pre_s/W, pre_s/W,
            inf_s, 100*inf_s/W, inf_s/W,
            drw_s, 100*drw_s/W, drw_s/W,
            cmp_s, 100*cmp_s/W, cmp_s/W,
            out_s, 100*out_s/W, out_s/W,
            gst_s, 100*gst_s/W, gst_s/W,
            enc_s, 100*enc_s/W, enc_s/W,
            infer_side, infer_side/W,
            (infer_side+display_side) > 0 ? 100*infer_side/(infer_side+display_side) : 0.0,
            display_side, display_side/W,
            (infer_side+display_side) > 0 ? 100*display_side/(infer_side+display_side) : 0.0,
            dec_s, enc_s,
            display_side > 1e-6 ? infer_side/display_side : 0.0);
    }

    free_worker_bufs(wb, in_size);
    for (size_t k = 0; k < sidecar_out_fds.size(); ++k) {
        if (sidecar_out_ptr[k]) munmap(sidecar_out_ptr[k], sidecar_out_alloc[k]);
        if (sidecar_out_fds[k] >= 0) ::close(sidecar_out_fds[k]);
    }
    axr_destroy(AXR_OBJECT(ctx));
    return 0;
}
