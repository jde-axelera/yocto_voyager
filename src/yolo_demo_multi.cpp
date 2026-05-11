// Multi-threaded, multi-stream Yolo11n end-to-end demo on Axelera Metis.
// Pipeline:
//   ffmpeg reader -> raw_q --(preproc)--> inst_q[w] x4 --(infer)--> done_q -> (reorder+draw) -> wr_q -> ffmpeg writer
//
// One axrModelInstance per Metis sub-device (4 total). Frames flow round-robin
// through 4 worker threads; a reorder thread emits results in the original order.
//
// Build: same CMakeLists.txt as yolo_demo.
// Run:   ./yolo_demo_mt <model.axm> <input.mp4> <output.mp4> [conf] [iou] [workers]

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <map>
#include <memory>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <signal.h>

#include "axruntime/axruntime.h"
#include "font8x8.h"
#include "coco_names.h"
// ======= pretty additions =======
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb/stb_truetype.h"
#include "liberation_sans_bold.h"
#include "coco_palette.h"

#include <map>
#include <mutex>

struct Glyph { int w=0,h=0,x0=0,y0=0,advance_px=0; std::vector<uint8_t> alpha; };
struct FontAtlas { std::map<char,Glyph> glyphs; int line_height=0, baseline=0; };
static FontAtlas* g_font14 = nullptr;
static FontAtlas* g_font18 = nullptr;

static FontAtlas* build_atlas(float pixel_height) {
    static stbtt_fontinfo fi;
    static bool inited = false;
    static std::mutex init_mu;
    {
        std::lock_guard<std::mutex> lk(init_mu);
        if (!inited) {
            if (!stbtt_InitFont(&fi, kLiberationSansBold,
                                stbtt_GetFontOffsetForIndex(kLiberationSansBold, 0))) {
                return nullptr;
            }
            inited = true;
        }
    }
    auto* a = new FontAtlas();
    float scale = stbtt_ScaleForPixelHeight(&fi, pixel_height);
    int ascent, descent, line_gap;
    stbtt_GetFontVMetrics(&fi, &ascent, &descent, &line_gap);
    a->baseline = (int)(ascent * scale);
    a->line_height = (int)((ascent - descent + line_gap) * scale);
    for (int c = 0x20; c < 0x7F; ++c) {
        Glyph g{};
        int adv, lsb;
        stbtt_GetCodepointHMetrics(&fi, c, &adv, &lsb);
        g.advance_px = (int)std::round(adv * scale);
        int x0,y0,x1,y1;
        stbtt_GetCodepointBitmapBox(&fi, c, scale, scale, &x0, &y0, &x1, &y1);
        g.w = x1-x0; g.h = y1-y0; g.x0 = x0; g.y0 = y0;
        g.alpha.assign((size_t)g.w * g.h, 0);
        if (g.w > 0 && g.h > 0) {
            stbtt_MakeCodepointBitmap(&fi, g.alpha.data(), g.w, g.h, g.w, scale, scale, c);
        }
        a->glyphs[(char)c] = std::move(g);
    }
    return a;
}
static inline void blit_glyph_bgr(uint8_t* dst, int dw, int dh, const Glyph& g,
                                  int x_pen, int y_pen, uint8_t r, uint8_t gr, uint8_t b) {
    int gy0 = y_pen + g.y0;
    int gx0 = x_pen + g.x0;
    for (int j=0;j<g.h;++j) {
        int yy = gy0 + j;
        if ((unsigned)yy >= (unsigned)dh) continue;
        for (int i=0;i<g.w;++i) {
            int xx = gx0 + i;
            if ((unsigned)xx >= (unsigned)dw) continue;
            uint8_t a = g.alpha[j*g.w + i];
            if (!a) continue;
            uint8_t* px = dst + (yy*dw + xx)*3;
            px[0] = (uint8_t)((px[0]*(255-a) + b *a + 127)/255);
            px[1] = (uint8_t)((px[1]*(255-a) + gr*a + 127)/255);
            px[2] = (uint8_t)((px[2]*(255-a) + r *a + 127)/255);
        }
    }
}
static inline int text_width_(const FontAtlas& a, const char* s) {
    int w=0;
    for (; *s; ++s) {
        auto it = a.glyphs.find(*s);
        if (it != a.glyphs.end()) w += it->second.advance_px;
    }
    return w;
}
static inline void draw_text_ttf(uint8_t* dst, int dw, int dh, int x, int y, const char* s,
                                 uint8_t r, uint8_t gr, uint8_t b, const FontAtlas& atlas) {
    int cur_x = x;
    int baseline_y = y + atlas.baseline;
    for (; *s; ++s) {
        auto it = atlas.glyphs.find(*s);
        if (it == atlas.glyphs.end()) continue;
        blit_glyph_bgr(dst, dw, dh, it->second, cur_x, baseline_y, r, gr, b);
        cur_x += it->second.advance_px;
    }
}
static inline void draw_text_shadow(uint8_t* dst, int dw, int dh, int x, int y, const char* s,
                                    uint8_t r, uint8_t gr, uint8_t b, const FontAtlas& atlas) {
    draw_text_ttf(dst, dw, dh, x+1, y+1, s, 0, 0, 0, atlas);
    draw_text_ttf(dst, dw, dh, x,   y,   s, r, gr, b, atlas);
}
static inline void fill_rect_alpha(uint8_t* dst, int dw, int dh,
                                   int x1, int y1, int x2, int y2,
                                   uint8_t r, uint8_t gr, uint8_t b, uint8_t a) {
    if (x1>x2) std::swap(x1,x2);
    if (y1>y2) std::swap(y1,y2);
    x1 = std::max(0,x1); y1 = std::max(0,y1);
    x2 = std::min(dw-1,x2); y2 = std::min(dh-1,y2);
    for (int yy=y1; yy<=y2; ++yy) {
        uint8_t* p = dst + (yy*dw + x1)*3;
        for (int xx=x1; xx<=x2; ++xx, p+=3) {
            p[0] = (uint8_t)((p[0]*(255-a) + b *a + 127)/255);
            p[1] = (uint8_t)((p[1]*(255-a) + gr*a + 127)/255);
            p[2] = (uint8_t)((p[2]*(255-a) + r *a + 127)/255);
        }
    }
}


using clk = std::chrono::steady_clock;

// configured per-stream target FPS (defaults to 25; overridable via the last CLI arg)
static int g_target_fps = 25;

static std::atomic<bool> g_shutdown{false};
static void on_sig(int) { g_shutdown.store(true); }

// =====================================================================
// dma-heap + dma-buf ABI (subset; SBC lacks linux/dma-heap.h header)
// =====================================================================
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/types.h>

struct dma_heap_allocation_data {
    __u64 len;
    __u32 fd;
    __u32 fd_flags;
    __u64 heap_flags;
};
#define DMA_HEAP_IOCTL_ALLOC _IOWR('H', 0x0, struct dma_heap_allocation_data)

struct dma_buf_sync {
    __u64 flags;
};
#define DMA_BUF_SYNC_READ    (1u << 0)
#define DMA_BUF_SYNC_WRITE   (2u << 0)
#define DMA_BUF_SYNC_START   (0u << 2)
#define DMA_BUF_SYNC_END     (1u << 2)
#define DMA_BUF_IOCTL_SYNC   _IOW('b', 0, struct dma_buf_sync)

class InputBufferPool {
public:
    struct Buf { int fd = -1; void* ptr = nullptr; size_t size = 0; };
    InputBufferPool(size_t buf_size, int count) {
        int heap_fd = ::open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
        if (heap_fd < 0) { std::perror("open(/dev/dma_heap/system)"); std::exit(1); }
        bufs_.resize(count);
        for (int i = 0; i < count; ++i) {
            dma_heap_allocation_data a{};
            a.len = buf_size;
            a.fd_flags = O_RDWR | O_CLOEXEC;
            if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &a) < 0) {
                std::perror("DMA_HEAP_IOCTL_ALLOC"); std::exit(1);
            }
            void* p = mmap(nullptr, buf_size, PROT_READ | PROT_WRITE, MAP_SHARED, (int)a.fd, 0);
            if (p == MAP_FAILED) { std::perror("mmap dmabuf"); std::exit(1); }
            bufs_[i] = { (int)a.fd, p, buf_size };
            free_.push(i);
        }
        ::close(heap_fd);
    }
    ~InputBufferPool() {
        for (auto& b : bufs_) {
            if (b.ptr) munmap(b.ptr, b.size);
            if (b.fd >= 0) ::close(b.fd);
        }
    }
    int acquire() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&]{ return !free_.empty(); });
        int i = free_.front(); free_.pop();
        return i;
    }
    void release(int i) {
        { std::lock_guard<std::mutex> lk(mu_); free_.push(i); }
        cv_.notify_one();
    }
    const Buf& operator[](int i) const { return bufs_[i]; }
    static void sync_start_write(int fd) {
        dma_buf_sync s{ DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE };
        ioctl(fd, DMA_BUF_IOCTL_SYNC, &s);
    }
    static void sync_end_write(int fd) {
        dma_buf_sync s{ DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE };
        ioctl(fd, DMA_BUF_IOCTL_SYNC, &s);
    }
private:
    std::vector<Buf> bufs_;
    std::queue<int> free_;
    std::mutex mu_;
    std::condition_variable cv_;
};


// =====================================================================
// Drawing primitives (BGR24)
// =====================================================================
struct Image { uint8_t* data; int w; int h; };

static inline void put_px(const Image& im, int x, int y, uint8_t b, uint8_t g, uint8_t r) {
    if ((unsigned)x >= (unsigned)im.w || (unsigned)y >= (unsigned)im.h) return;
    uint8_t* p = im.data + (y*im.w + x) * 3;
    p[0]=b; p[1]=g; p[2]=r;
}
static void draw_hline(const Image& im, int x1, int x2, int y, uint8_t b, uint8_t g, uint8_t r) {
    if (y < 0 || y >= im.h) return;
    if (x1 > x2) std::swap(x1, x2);
    x1 = std::max(0, x1); x2 = std::min(im.w-1, x2);
    uint8_t* p = im.data + (y*im.w + x1) * 3;
    for (int x = x1; x <= x2; ++x, p += 3) { p[0]=b; p[1]=g; p[2]=r; }
}
static void draw_vline(const Image& im, int x, int y1, int y2, uint8_t b, uint8_t g, uint8_t r) {
    if (x < 0 || x >= im.w) return;
    if (y1 > y2) std::swap(y1, y2);
    y1 = std::max(0, y1); y2 = std::min(im.h-1, y2);
    for (int y = y1; y <= y2; ++y) {
        uint8_t* p = im.data + (y*im.w + x) * 3;
        p[0]=b; p[1]=g; p[2]=r;
    }
}
static void draw_rect(const Image& im, int x1, int y1, int x2, int y2,
                      uint8_t b, uint8_t g, uint8_t r, int thick = 2) {
    for (int t = 0; t < thick; ++t) {
        draw_hline(im, x1, x2, y1 + t, b, g, r);
        draw_hline(im, x1, x2, y2 - t, b, g, r);
        draw_vline(im, x1 + t, y1, y2, b, g, r);
        draw_vline(im, x2 - t, y1, y2, b, g, r);
    }
}
static void fill_rect(const Image& im, int x1, int y1, int x2, int y2,
                      uint8_t b, uint8_t g, uint8_t r) {
    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);
    x1 = std::max(0, x1); y1 = std::max(0, y1);
    x2 = std::min(im.w-1, x2); y2 = std::min(im.h-1, y2);
    for (int y = y1; y <= y2; ++y) {
        uint8_t* p = im.data + (y*im.w + x1) * 3;
        for (int x = x1; x <= x2; ++x, p += 3) { p[0]=b; p[1]=g; p[2]=r; }
    }
}
static void put_char(const Image& im, int x, int y, char c,
                     uint8_t b, uint8_t g, uint8_t r, int s = 1) {
    if (c < 0x20 || c > 0x7F) c = '?';
    const uint8_t* glyph = FONT8X8[c - 0x20];
    for (int gy = 0; gy < 8; ++gy) {
        uint8_t row = glyph[gy];
        for (int gx = 0; gx < 8; ++gx) {
            if (row & (1 << gx)) {
                for (int sy = 0; sy < s; ++sy)
                    for (int sx = 0; sx < s; ++sx)
                        put_px(im, x + gx*s + sx, y + gy*s + sy, b, g, r);
            }
        }
    }
}
static void put_text(const Image& im, int x, int y, const char* s,
                     uint8_t b, uint8_t g, uint8_t r, int scale = 1) {
    while (*s) { put_char(im, x, y, *s, b, g, r, scale); x += 8*scale; ++s; }
}
static inline void class_color(int cls, uint8_t& b, uint8_t& g, uint8_t& r) {
    if (cls < 0 || cls >= 80) cls = 0;
    r = COCO_PALETTE[cls][0]; g = COCO_PALETTE[cls][1]; b = COCO_PALETTE[cls][2];
}

// =====================================================================
// ffmpeg subprocess
// =====================================================================
struct Subprocess { pid_t pid = -1; int fd = -1; };

static Subprocess ffmpeg_reader(const std::string& path, int& w, int& h, double& fps, int64_t& nframes) {
    char cmd[1024];
    std::snprintf(cmd, sizeof cmd,
        "ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames "
        "-of default=nw=1 '%s'", path.c_str());
    FILE* p = popen(cmd, "r");
    if (!p) { std::perror("popen ffprobe"); std::exit(1); }
    char line[256];
    w = 0; h = 0; fps = 0; nframes = 0;
    while (std::fgets(line, sizeof line, p)) {
        int a, b;
        if (std::sscanf(line, "width=%d", &a) == 1) w = a;
        else if (std::sscanf(line, "height=%d", &a) == 1) h = a;
        else if (std::sscanf(line, "r_frame_rate=%d/%d", &a, &b) == 2 && b) fps = (double)a / b;
        else if (std::sscanf(line, "nb_frames=%lld", (long long*)&nframes) == 1) {}
    }
    pclose(p);
    if (w <= 0 || h <= 0) { std::fprintf(stderr, "ffprobe: bad dimensions\n"); std::exit(1); }

    int pipefd[2];
    if (pipe(pipefd) < 0) { std::perror("pipe"); std::exit(1); }
    pid_t pid = fork();
    if (pid == 0) {
        ::close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        ::close(pipefd[1]);
        char rate_arg[32];
        std::snprintf(rate_arg, sizeof rate_arg, "%d", g_target_fps);
        execlp("ffmpeg",
               "ffmpeg", "-loglevel", "error",
               "-stream_loop", "-1",
               "-re",                                  // pace input at native rate
               "-i", path.c_str(),
               "-r", rate_arg,                         // emit at target fps (drop/dup as needed)
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
               (char*)nullptr);
        _exit(127);
    }
    ::close(pipefd[1]);
    return Subprocess{pid, pipefd[0]};
}
static Subprocess ffmpeg_writer(const std::string& path, int w, int h, double fps) {
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


// TCP MPEG-TS H.264 streamer (h264_rkmpp hw encoder, listen on a TCP port).
// The display fd is set non-blocking by the caller so the writer thread can drop
// frames if no viewer is connected yet.
static Subprocess ffmpeg_tcp_streamer(int w, int h, double fps, int port) {
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
               "ffmpeg",
               "-loglevel", "warning",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", size, "-r", rate, "-i", "-",
               "-c:v", "h264_rkmpp", "-pix_fmt", "nv12",
               "-f", "mpegts",
               url,
               (char*)nullptr);
        _exit(127);
    }
    ::close(pipefd[0]);
    return Subprocess{pid, pipefd[1]};
}

static Subprocess ffplay_display(int w, int h, double fps, const char* title) {
    int pipefd[2];
    if (pipe(pipefd) < 0) { std::perror("pipe"); std::exit(1); }
    pid_t pid = fork();
    if (pid == 0) {
        ::close(pipefd[1]);
        dup2(pipefd[0], STDIN_FILENO);
        ::close(pipefd[0]);
        // gst-launch needs each property as its own argv token. Build the argv programmatically.
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
    (void)title;  // gst-launch doesn't take a window title; user can pass through env
    return Subprocess{pid, pipefd[1]};
}

static bool read_full(int fd, void* buf, size_t n) {
    auto* p = (uint8_t*)buf;
    while (n) { ssize_t r = ::read(fd, p, n); if (r <= 0) return false; p += r; n -= r; }
    return true;
}
static bool write_full(int fd, const void* buf, size_t n) {
    auto* p = (const uint8_t*)buf;
    while (n) { ssize_t r = ::write(fd, p, n); if (r <= 0) return false; p += r; n -= r; }
    return true;
}
static void close_sub(Subprocess& s) {
    if (s.fd >= 0) ::close(s.fd);
    if (s.pid > 0) {
        // Give the child up to 2 seconds to exit cleanly; then SIGTERM, then SIGKILL.
        int st = 0;
        for (int i = 0; i < 20; ++i) {
            pid_t r = waitpid(s.pid, &st, WNOHANG);
            if (r == s.pid) { s.pid = -1; s.fd = -1; return; }
            if (r < 0) break;
            usleep(100*1000);
        }
        kill(s.pid, SIGTERM);
        for (int i = 0; i < 10; ++i) {
            pid_t r = waitpid(s.pid, &st, WNOHANG);
            if (r == s.pid) { s.pid = -1; s.fd = -1; return; }
            usleep(100*1000);
        }
        kill(s.pid, SIGKILL);
        waitpid(s.pid, &st, 0);
    }
    s.fd = -1; s.pid = -1;
}

// =====================================================================
// YOLO preprocess + postprocess (same as v1)
// =====================================================================
struct PreprocCtx {
    int H, W, channels;
    int yp_l, yp_r, xp_l, xp_r, cp_l, cp_r;
    int uw, uh;
    int8_t pad_val;
};
static PreprocCtx make_preproc(const axrTensorInfo& info) {
    PreprocCtx c{};
    c.H = (int)info.dims[1];
    c.W = (int)info.dims[2];
    c.channels = (int)info.dims[3];
    c.yp_l = info.padding[1][0]; c.yp_r = info.padding[1][1];
    c.xp_l = info.padding[2][0]; c.xp_r = info.padding[2][1];
    c.cp_l = info.padding[3][0]; c.cp_r = info.padding[3][1];
    c.uh = c.H - c.yp_l - c.yp_r;
    c.uw = c.W - c.xp_l - c.xp_r;
    c.pad_val = (int8_t)std::clamp(info.zero_point, -128, 127);
    return c;
}
static void preprocess(const uint8_t* src_bgr, int sw, int sh,
                       int8_t* dst, const PreprocCtx& c,
                       float& letter_scale, int& padx_640, int& pady_640) {
    const int channels = c.channels;
    const int duw = c.uw, duh = c.uh;
    float sx = (float)duw / sw, sy = (float)duh / sh;
    letter_scale = std::min(sx, sy);
    int new_w = (int)std::round(sw * letter_scale);
    int new_h = (int)std::round(sh * letter_scale);
    padx_640 = (duw - new_w) / 2;
    pady_640 = (duh - new_h) / 2;

    std::fill_n(dst, c.yp_l * c.W * channels, c.pad_val);
    int8_t* row_out = dst + c.yp_l * c.W * channels;
    const uint32_t fp_x_step = (uint32_t)((double)sw * 65536.0 / new_w);
    const uint32_t fp_y_step = (uint32_t)((double)sh * 65536.0 / new_h);

    for (int y = 0; y < duh; ++y) {
        int8_t* row = row_out + y * c.W * channels;
        std::fill_n(row, c.xp_l * channels, c.pad_val);
        int8_t* p = row + c.xp_l * channels;
        if (y < pady_640 || y >= pady_640 + new_h) {
            std::fill_n(p, duw * channels, c.pad_val);
        } else {
            std::fill_n(p, padx_640 * channels, c.pad_val); p += padx_640 * channels;
            int src_y = (int)((uint64_t)(y - pady_640) * fp_y_step >> 16);
            if (src_y >= sh) src_y = sh - 1;
            const uint8_t* src_row = src_bgr + src_y * sw * 3;
            uint32_t fp_x = 0;
            for (int x = 0; x < new_w; ++x, fp_x += fp_x_step) {
                int src_x = fp_x >> 16;
                if (src_x >= sw) src_x = sw - 1;
                const uint8_t* px = src_row + src_x * 3;
                for (int k = 0; k < c.cp_l; ++k) *p++ = c.pad_val;
                *p++ = (int8_t)((int)px[2] - 128);
                *p++ = (int8_t)((int)px[1] - 128);
                *p++ = (int8_t)((int)px[0] - 128);
                for (int k = 0; k < c.cp_r; ++k) *p++ = c.pad_val;
            }
            int rest = duw - padx_640 - new_w;
            if (rest > 0) std::fill_n(p, rest * channels, c.pad_val);
        }
        std::fill_n(row + (c.xp_l + duw) * channels, c.xp_r * channels, c.pad_val);
    }
    std::fill_n(row_out + duh * c.W * channels, c.yp_r * c.W * channels, c.pad_val);
}

struct Detection { float x1, y1, x2, y2; float score; int cls; };
static std::array<std::array<int, 2>, 3> classify_outputs(const std::vector<axrTensorInfo>& outs) {
    std::array<std::array<int, 2>, 3> table{};
    for (auto& row : table) row = {-1, -1};
    for (size_t i = 0; i < outs.size(); ++i) {
        const auto& o = outs[i];
        int gh = (int)o.dims[1];
        int kind = (int)o.dims[3] == 64 ? 0 : 1;
        int s = (gh == 80) ? 0 : (gh == 40) ? 1 : 2;
        table[s][kind] = (int)i;
    }
    return table;
}
static void decode_dfl_sigmoid_filter(
    const std::vector<const int8_t*>& bufs,
    const std::vector<axrTensorInfo>& infos,
    const std::array<std::array<int, 2>, 3>& tbl,
    float conf_thresh,
    float scale_letterbox, int padx_640, int pady_640,
    int orig_w, int orig_h,
    std::vector<Detection>& out)
{
    out.clear();
    static const int strides[3] = {8, 16, 32};
    float bins[16];
    for (int s = 0; s < 3; ++s) {
        int bbox_idx = tbl[s][0], cls_idx = tbl[s][1];
        if (bbox_idx < 0 || cls_idx < 0) continue;
        const axrTensorInfo& bi = infos[bbox_idx];
        const axrTensorInfo& ci = infos[cls_idx];
        const int gh = (int)bi.dims[1], gw = (int)bi.dims[2];
        const int stride = strides[s];
        const int cls_ch_total = (int)ci.dims[3];
        const int cls_unpad_l = ci.padding[3][0];
        const int cls_unpad_n = cls_ch_total - cls_unpad_l - ci.padding[3][1];
        const int bbox_ch_total = (int)bi.dims[3];
        const float bs = bi.scale, bz = (float)bi.zero_point;
        const float cs = ci.scale, cz = (float)ci.zero_point;
        const int8_t* bbox_data = bufs[bbox_idx];
        const int8_t* cls_data  = bufs[cls_idx];

        for (int gy = 0; gy < gh; ++gy) {
            for (int gx = 0; gx < gw; ++gx) {
                const int8_t* bbox_q = bbox_data + ((gy*gw + gx) * bbox_ch_total);
                const int8_t* cls_q  = cls_data  + ((gy*gw + gx) * cls_ch_total) + cls_unpad_l;
                int best_cls = -1; float best_logit = -1e30f;
                for (int k = 0; k < cls_unpad_n; ++k) {
                    float f = ((int)cls_q[k] - cz) * cs;
                    if (f > best_logit) { best_logit = f; best_cls = k; }
                }
                float best_score = 1.0f / (1.0f + std::exp(-best_logit));
                if (best_score < conf_thresh) continue;
                float dist[4];
                for (int side = 0; side < 4; ++side) {
                    float maxv = -1e30f;
                    for (int b = 0; b < 16; ++b) {
                        float f = ((int)bbox_q[side*16 + b] - bz) * bs;
                        bins[b] = f;
                        if (f > maxv) maxv = f;
                    }
                    float sum = 0;
                    for (int b = 0; b < 16; ++b) { bins[b] = std::exp(bins[b] - maxv); sum += bins[b]; }
                    float inv = 1.0f / sum, exp_val = 0;
                    for (int b = 0; b < 16; ++b) exp_val += bins[b] * inv * (float)b;
                    dist[side] = exp_val;
                }
                float cx = (gx + 0.5f) * stride, cy = (gy + 0.5f) * stride;
                float x1 = cx - dist[0]*stride, y1 = cy - dist[1]*stride;
                float x2 = cx + dist[2]*stride, y2 = cy + dist[3]*stride;
                x1 = (x1 - padx_640) / scale_letterbox;
                y1 = (y1 - pady_640) / scale_letterbox;
                x2 = (x2 - padx_640) / scale_letterbox;
                y2 = (y2 - pady_640) / scale_letterbox;
                x1 = std::clamp(x1, 0.0f, (float)orig_w - 1);
                y1 = std::clamp(y1, 0.0f, (float)orig_h - 1);
                x2 = std::clamp(x2, 0.0f, (float)orig_w - 1);
                y2 = std::clamp(y2, 0.0f, (float)orig_h - 1);
                if (x2 <= x1 || y2 <= y1) continue;
                out.push_back({x1, y1, x2, y2, best_score, best_cls});
            }
        }
    }
}
static float iou(const Detection& a, const Detection& b) {
    float xx1 = std::max(a.x1, b.x1), yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2), yy2 = std::min(a.y2, b.y2);
    float iw = std::max(0.0f, xx2 - xx1), ih = std::max(0.0f, yy2 - yy1);
    float inter = iw * ih;
    float ua = (a.x2 - a.x1) * (a.y2 - a.y1);
    float ub = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (ua + ub - inter + 1e-9f);
}
static std::vector<Detection> nms(std::vector<Detection> dets, float iou_thresh, int max_out = 300) {
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> out;
    out.reserve(std::min<size_t>(dets.size(), max_out));
    std::vector<char> killed(dets.size(), 0);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (killed[i]) continue;
        out.push_back(dets[i]);
        if ((int)out.size() >= max_out) break;
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (killed[j] || dets[j].cls != dets[i].cls) continue;
            if (iou(dets[i], dets[j]) > iou_thresh) killed[j] = 1;
        }
    }
    return out;
}

// =====================================================================
// Bounded thread-safe queue
// =====================================================================
template<class T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t cap) : cap_(cap) {}
    void push(T&& v) {
        std::unique_lock<std::mutex> lk(mu_);
        not_full_.wait(lk, [&]{ return q_.size() < cap_ || closed_; });
        if (closed_) return;
        q_.push(std::move(v));
        not_empty_.notify_one();
    }
    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(mu_);
        not_empty_.wait(lk, [&]{ return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop();
        not_full_.notify_one();
        return true;
    }
    void close() {
        std::lock_guard<std::mutex> lk(mu_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }
private:
    std::queue<T> q_;
    size_t cap_;
    std::mutex mu_;
    std::condition_variable not_empty_, not_full_;
    bool closed_ = false;
};


// 1-deep leaky slot. Newer pushes overwrite older. Used to keep one current frame
// available for the display thread; the inference pipeline never stalls if no viewer
// is connected or the viewer is slow.
template<class T>
class LeakyOne {
    std::mutex m_;
    std::condition_variable cv_;
    std::unique_ptr<T> slot_;
    bool closed_ = false;
public:
    void push(T&& v) {
        std::lock_guard<std::mutex> lk(m_);
        if (closed_) return;
        slot_ = std::make_unique<T>(std::move(v));   // overwrites
        cv_.notify_one();
    }
    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&]{ return slot_ || closed_; });
        if (!slot_) return false;
        out = std::move(*slot_);
        slot_.reset();
        return true;
    }
    void close() {
        std::lock_guard<std::mutex> lk(m_);
        closed_ = true;
        cv_.notify_all();
    }
};

// =====================================================================
// Per-frame payload
// =====================================================================
struct Frame {
    int idx = -1;
    int stream_id = 0;
    int sw = 0, sh = 0;
    std::vector<uint8_t> bgr;          // original frame
    std::vector<int8_t> input;          // preprocessed (matches model input layout)
    std::vector<std::vector<int8_t>> outputs;  // one per output tensor
    std::vector<axrArgument> in_args, out_args;
    float lscale = 1.0f;
    int padx = 0, pady = 0;
    int in_buf_idx = -1;
};

using FramePtr = std::unique_ptr<Frame>;

// Per-stream descriptor.
struct Stream {
    int id;
    std::string in_path;
    std::string out_path;
    int sw = 0, sh = 0;
    double fps_in = 30.0;
    int64_t nframes = 0;
    Subprocess reader{}, writer{};
    BoundedQueue<FramePtr>* write_q = nullptr;
    std::thread writer_thread;
    // per-stream reorder state used by the drawer
    std::map<int, FramePtr> pending;
    int next_idx = 0;
    // metrics
    std::atomic<int> decoded{0}, drawn{0}, written{0};
    // most-recent annotated frame snapshot for composite display
    std::mutex snap_mu;
    std::vector<uint8_t> snapshot;
};


// =====================================================================
// Main
// =====================================================================
int main(int argc, char** argv) {
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa{}; sa.sa_handler = on_sig;
    sigaction(SIGINT, &sa, nullptr); sigaction(SIGTERM, &sa, nullptr);
    if (argc < 4) {
        std::fprintf(stderr,
            "Usage: %s <model.json|.axm> <video1.mp4[,video2.mp4,...]> <output_prefix> "
            "[conf=0.25] [iou=0.45] [workers=1] [bench=0] [dmabuf=0] [preproc=4] [display=0] [fps=25]\n"
            "  - up to 10 input videos comma-separated; each paced to fps (default 25).\n"
            "  - outputs go to <output_prefix>_0.mp4, <output_prefix>_1.mp4, ...\n"
            "  - display=1: composite-grid X11 window (DISPLAY=:0).\n"
            "  - display=2: composite-grid TCP MPEG-TS stream on :5000 (auto-respawn).\n",
            argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    std::string in_arg     = argv[2];
    std::string out_prefix = argv[3];
    float conf_thresh = (argc > 4) ? std::atof(argv[4]) : 0.25f;
    float iou_thresh  = (argc > 5) ? std::atof(argv[5]) : 0.45f;
    int N             = (argc > 6) ? std::atoi(argv[6]) : 1;
    if (N < 1) N = 1;
    int bench         = (argc > 7) ? std::atoi(argv[7]) : 0;
    int use_dmabuf    = (argc > 8) ? std::atoi(argv[8]) : 0;
    int preproc_threads = (argc > 9) ? std::atoi(argv[9]) : 4;
    if (preproc_threads < 1) preproc_threads = 1;
    int live_display  = (argc > 10) ? std::atoi(argv[10]) : 0;
    g_target_fps      = (argc > 11) ? std::atoi(argv[11]) : 25;
    if (g_target_fps < 1) g_target_fps = 25;

    g_font14 = build_atlas(14.0f);
    g_font18 = build_atlas(18.0f);

    // ---- Parse comma-separated input paths into a vector of Streams ----
    std::vector<std::unique_ptr<Stream>> streams;
    {
        std::string buf; buf.reserve(in_arg.size());
        for (size_t i = 0; i <= in_arg.size(); ++i) {
            char c = (i == in_arg.size()) ? ',' : in_arg[i];
            if (c == ',') {
                if (!buf.empty()) {
                    auto s = std::make_unique<Stream>();
                    s->id = (int)streams.size();
                    s->in_path = buf;
                    char op[512];
                    std::snprintf(op, sizeof op, "%s_%d.mp4", out_prefix.c_str(), s->id);
                    s->out_path = op;
                    streams.push_back(std::move(s));
                    buf.clear();
                }
            } else { buf.push_back(c); }
        }
    }
    if (streams.empty() || streams.size() > 10) {
        std::fprintf(stderr, "ERROR: need 1..10 input videos (got %zu)\n", streams.size());
        return 1;
    }

    // ---- axruntime: 1 context/model/connection/instance (single worker; batch=4 model) ----
    axrContext* ctx = axr_create_context();
    if (!ctx) { std::fprintf(stderr, "axr_create_context failed\n"); return 1; }
    axrModel* model = axr_load_model(ctx, model_path);
    if (!model) {
        std::fprintf(stderr, "load_model: %s\n", axr_last_error_string(AXR_OBJECT(ctx)));
        return 1;
    }
    size_t n_in = axr_num_model_inputs(model);
    size_t n_out = axr_num_model_outputs(model);
    std::vector<axrTensorInfo> in_infos(n_in), out_infos(n_out);
    for (size_t i = 0; i < n_in; ++i)  in_infos[i]  = axr_get_model_input(model, i);
    for (size_t i = 0; i < n_out; ++i) out_infos[i] = axr_get_model_output(model, i);
    if (n_in != 1) { std::fprintf(stderr, "expected 1 input\n"); return 1; }
    int batch = (int)in_infos[0].dims[0];

    axrConnection* conn = axr_device_connect(ctx, nullptr, batch * N, nullptr);
    if (!conn) {
        std::fprintf(stderr, "device_connect: %s\n", axr_last_error_string(AXR_OBJECT(ctx)));
        return 1;
    }
    int input_dmabuf_eff = 1;  // b4-style: worker always packs into a worker-owned dmabuf
    (void)use_dmabuf;
    std::string props = "input_dmabuf=" + std::to_string(input_dmabuf_eff)
                      + ";num_sub_devices=" + std::to_string(batch)
                      + ";aipu_cores="      + std::to_string(batch)
                      + ";double_buffer=1";
    axrProperties* properties = axr_create_properties(ctx, props.c_str());
    std::vector<axrModelInstance*> insts(N);
    for (int i = 0; i < N; ++i) {
        insts[i] = axr_load_model_instance(conn, model, properties);
        if (!insts[i]) {
            std::fprintf(stderr, "load_model_instance[%d]: %s\n", i,
                axr_last_error_string(AXR_OBJECT(ctx)));
            return 1;
        }
    }

    auto pre_ctx = make_preproc(in_infos[0]);
    auto out_tbl = classify_outputs(out_infos);
    const size_t in_size = axr_tensor_size(&in_infos[0]);
    std::vector<size_t> out_sizes(n_out);
    for (size_t i = 0; i < n_out; ++i) out_sizes[i] = axr_tensor_size(&out_infos[i]);

    // ---- Per-stream ffmpeg I/O ----
    int common_sw = 0, common_sh = 0;
    double common_fps = (double)g_target_fps;
    for (auto& s : streams) {
        // ffprobe
        int64_t nf = 0;
        s->reader = ffmpeg_reader(s->in_path, s->sw, s->sh, s->fps_in, nf);
        s->nframes = nf;
        if (common_sw == 0) { common_sw = s->sw; common_sh = s->sh; }
        else if (common_sw != s->sw || common_sh != s->sh) {
            std::fprintf(stderr,
                "ERROR: stream %d (%dx%d) does not match stream 0 (%dx%d). "
                "All inputs must share the same resolution.\n",
                s->id, s->sw, s->sh, common_sw, common_sh);
            return 1;
        }
        s->writer = ffmpeg_writer(s->out_path, s->sw, s->sh, common_fps);
        s->snapshot.assign((size_t)s->sw * s->sh * 3, 0);
        s->write_q = new BoundedQueue<FramePtr>(8);
    }

    // ---- HUD/summary banner ----
    std::fprintf(stderr,
        "model    : %s\n"
        "streams  : %zu  (resolution %dx%d, target %d fps each, aggregate %.0f fps)\n",
        model_path, streams.size(), common_sw, common_sh, g_target_fps,
        streams.size() * (double)g_target_fps);
    for (auto& s : streams) {
        std::fprintf(stderr, "  [%d] %s -> %s\n",
                     s->id, s->in_path.c_str(), s->out_path.c_str());
    }
    std::fprintf(stderr, "conf=%.2f  iou=%.2f  preproc=%d  bench=%d  display=%d\n"
                         "props    : %s\n\n",
                 conf_thresh, iou_thresh, preproc_threads, bench, live_display, props.c_str());

    // ---- Queues + instance dmabufs (b4-style) ----
    BoundedQueue<FramePtr> raw_q(64);
    std::vector<std::unique_ptr<BoundedQueue<FramePtr>>> inst_q(N);
    for (int i = 0; i < N; ++i) inst_q[i] = std::make_unique<BoundedQueue<FramePtr>>(16);
    BoundedQueue<FramePtr> done_q(64);

    // ---- Worker batch dmabufs (one per inference worker, sized to batch input bytes) ----
    struct WorkerBufs {
        int in_fd = -1;
        void* in_ptr = nullptr;
        std::vector<std::vector<int8_t>> out_heap;
    };
    std::vector<WorkerBufs> wb(N);
    {
        int heap_fd = ::open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
        if (heap_fd < 0) { std::perror("open dma_heap/system"); std::exit(1); }
        for (int i = 0; i < N; ++i) {
            dma_heap_allocation_data a{};
            a.len = in_size;
            a.fd_flags = O_RDWR | O_CLOEXEC;
            if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &a) < 0) { std::perror("alloc"); std::exit(1); }
            void* p = mmap(nullptr, in_size, PROT_READ|PROT_WRITE, MAP_SHARED, (int)a.fd, 0);
            if (p == MAP_FAILED) { std::perror("mmap"); std::exit(1); }
            wb[i] = { (int)a.fd, p, std::vector<std::vector<int8_t>>(n_out) };
            for (size_t k = 0; k < n_out; ++k) wb[i].out_heap[k].resize(out_sizes[k]);
        }
        ::close(heap_fd);
    }

    std::atomic<int64_t> infer_ns_total{0};
    std::atomic<int> frames_inferred{0};
    auto t0 = clk::now();

    // ---- Stats logger + shutdown watcher (logs throughput every 2 s; kills readers on signal) ----
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
        }
    });
    std::thread sig_watcher([&]() {
        while (!g_shutdown.load(std::memory_order_relaxed))
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        // shutdown: send SIGTERM to all reader subprocesses so read_full returns 0
        for (auto& sp : streams) {
            if (sp->reader.pid > 0) ::kill(sp->reader.pid, SIGTERM);
        }
    });

    // ---- Decoder threads (one per stream) ----
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
                if (!read_full(s->reader.fd, f->bgr.data(), frame_bytes)) break;
                f->idx = idx++;
                s->decoded.fetch_add(1, std::memory_order_relaxed);
                raw_q.push(std::move(f));
            }
        });
    }

    // ---- Preproc threads (shared raw_q) ----
    std::vector<std::thread> preprocs;
    std::atomic<int> preproc_done{0};
    std::atomic<bool> decoders_done{false};
    for (int t = 0; t < preproc_threads; ++t) {
        preprocs.emplace_back([&]() {
            FramePtr f;
            while (raw_q.pop(f)) {
                f->outputs.resize(n_out);
                f->in_args.assign(1, {});
                f->out_args.assign(n_out, {});
                f->input.assign(in_size / batch, 0);
                preprocess(f->bgr.data(), f->sw, f->sh,
                           reinterpret_cast<int8_t*>(f->input.data()),
                           pre_ctx, f->lscale, f->padx, f->pady);
                f->in_args[0].ptr = f->input.data();
                f->in_args[0].size = f->input.size();
                for (size_t k = 0; k < n_out; ++k) {
                    f->outputs[k].assign(out_sizes[k], 0);
                    f->out_args[k].ptr = f->outputs[k].data();
                    f->out_args[k].size = f->outputs[k].size();
                }
                int w = 0;  // single worker (b4 packing)
                inst_q[w]->push(std::move(f));
            }
            if (preproc_done.fetch_add(1) + 1 == preproc_threads) {
                for (auto& q : inst_q) q->close();
            }
        });
    }

    // ---- Worker thread (b4 packing of 4 frames per inference call) ----
    std::vector<std::thread> workers;
    for (int i = 0; i < N; ++i) {
        workers.emplace_back([&, i]() {
            const int B = batch;
            const size_t in_slot = in_size / B;
            std::vector<size_t> out_slot(n_out);
            for (size_t k = 0; k < n_out; ++k) out_slot[k] = out_sizes[k] / B;
            std::vector<axrArgument> in_args(1), out_args(n_out);
            in_args[0].ptr = nullptr;
            in_args[0].fd  = wb[i].in_fd;
            in_args[0].offset = 0;
            in_args[0].size = in_size;
            for (size_t k = 0; k < n_out; ++k) {
                out_args[k].ptr = wb[i].out_heap[k].data();
                out_args[k].size = wb[i].out_heap[k].size();
                out_args[k].fd = 0; out_args[k].offset = 0;
            }
            std::vector<FramePtr> batch_frames;
            batch_frames.reserve(B);
            auto run_batch = [&]() {
                if (batch_frames.empty()) return;
                InputBufferPool::sync_start_write(wb[i].in_fd);
                for (int b = 0; b < (int)batch_frames.size(); ++b) {
                    auto& f = batch_frames[b];
                    std::memcpy(static_cast<uint8_t*>(wb[i].in_ptr) + b * in_slot,
                                f->input.data(), in_slot);
                }
                for (int b = (int)batch_frames.size(); b < B; ++b) {
                    std::memcpy(static_cast<uint8_t*>(wb[i].in_ptr) + b * in_slot,
                                wb[i].in_ptr, in_slot);
                }
                InputBufferPool::sync_end_write(wb[i].in_fd);
                auto t0 = clk::now();
                axrResult r = axr_run_model_instance(insts[i],
                    in_args.data(), in_args.size(),
                    out_args.data(), out_args.size());
                auto t1 = clk::now();
                if (r != AXR_SUCCESS) {
                    std::fprintf(stderr, "[worker %d] run failed: %s\n", i,
                        axr_last_error_string(AXR_OBJECT(ctx)));
                    return;
                }
                infer_ns_total.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                    std::memory_order_relaxed);
                for (int b = 0; b < (int)batch_frames.size(); ++b) {
                    auto& f = batch_frames[b];
                    f->outputs.resize(n_out);
                    for (size_t k = 0; k < n_out; ++k) {
                        f->outputs[k].assign(out_slot[k], 0);
                        std::memcpy(f->outputs[k].data(),
                                    wb[i].out_heap[k].data() + b * out_slot[k],
                                    out_slot[k]);
                    }
                    frames_inferred.fetch_add(1, std::memory_order_relaxed);
                    done_q.push(std::move(f));
                }
                batch_frames.clear();
            };
            FramePtr f;
            while (inst_q[i]->pop(f)) {
                batch_frames.push_back(std::move(f));
                if ((int)batch_frames.size() == B) run_batch();
            }
            run_batch();
        });
    }

    // ---- Drawer thread: per-stream reorder + draw + push to per-stream write_q + snapshot ----
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
                if (bench < 2) {
                    std::vector<const int8_t*> ptrs(cur->outputs.size());
                    for (size_t k = 0; k < cur->outputs.size(); ++k) ptrs[k] = cur->outputs[k].data();
                    dets.clear();
                    decode_dfl_sigmoid_filter(ptrs, out_infos, out_tbl,
                                              conf_thresh, cur->lscale, cur->padx, cur->pady,
                                              cur->sw, cur->sh, dets);
                    auto kept = nms(std::move(dets), iou_thresh);
                    if (bench == 0) {
                        Image im{cur->bgr.data(), cur->sw, cur->sh};
                        FontAtlas* font = g_font14;
                        for (const auto& d : kept) {
                            uint8_t bc, gc, rc; class_color(d.cls, bc, gc, rc);
                            int x1=(int)d.x1,y1=(int)d.y1,x2=(int)d.x2,y2=(int)d.y2;
                            draw_rect(im, x1, y1, x2, y2, bc, gc, rc, 2);
                            char label[128];
                            std::snprintf(label, sizeof label, "%s %d%%",
                                          (d.cls>=0 && d.cls<80) ? COCO_NAMES[d.cls] : "?",
                                          (int)(d.score*100));
                            int tw = font ? text_width_(*font, label) : (int)std::strlen(label)*8;
                            int th = font ? font->line_height : 10;
                            int ph = 4, pv = 1;
                            int bw = tw + 2*ph, bh = th + 2*pv;
                            int ly0 = y1 - bh;
                            if (ly0 < 0) ly0 = y1 + 1;
                            fill_rect_alpha(im.data, im.w, im.h, x1, ly0, x1 + bw, ly0 + bh, rc, gc, bc, 230);
                            if (font) draw_text_ttf(im.data, im.w, im.h, x1 + ph, ly0 + pv, label, 255, 255, 255, *font);
                        }
                        // Per-stream HUD strip top-left
                        char hud[160];
                        double total_s = std::max(1e-6, std::chrono::duration<double>(clk::now()-t0).count());
                        double sfps = s->drawn.load(std::memory_order_relaxed) / total_s;
                        std::snprintf(hud, sizeof hud, "stream %d  %.1f fps  dets %zu",
                                      sid, sfps, kept.size());
                        FontAtlas* hf = g_font18;
                        int htw = hf ? text_width_(*hf, hud) : (int)std::strlen(hud)*8;
                        int hth = hf ? hf->line_height : 12;
                        fill_rect_alpha(im.data, im.w, im.h, 8, 8, 8 + htw + 12, 8 + hth + 4, 0, 0, 0, 170);
                        if (hf) draw_text_shadow(im.data, im.w, im.h, 14, 9, hud, 255, 255, 255, *hf);
                    }
                }
                s->drawn.fetch_add(1, std::memory_order_relaxed);
                // Snapshot for composite display (overwrite-newest)
                if (live_display) {
                    std::lock_guard<std::mutex> lk(s->snap_mu);
                    std::memcpy(s->snapshot.data(), cur->bgr.data(),
                                std::min(s->snapshot.size(), cur->bgr.size()));
                }
                s->write_q->push(std::move(cur));
            }
        }
        for (auto& sp : streams) sp->write_q->close();
    });

    // ---- Per-stream writer threads ----
    for (auto& sp : streams) {
        Stream* s = sp.get();
        s->writer_thread = std::thread([&, s]() {
            FramePtr f;
            while (s->write_q->pop(f)) {
                if (bench != 0) continue;
                if (!write_full(s->writer.fd, f->bgr.data(), f->bgr.size())) break;
                s->written.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // ---- Composite display: fixed 4x4 grid (up to 16 streams), unused cells stay black ----
    // Pipeline: producer thread (paced ~30 fps, decimates snapshots into composite, pushes to
    // leaky 1-slot) + consumer thread (blocking write to disp subprocess, respawn on failure).
    LeakyOne<std::vector<uint8_t>> disp_slot;
    std::thread disp_producer, disp_consumer;
    std::atomic<bool> disp_stop{false};
    const int GRID_COLS = 4, GRID_ROWS = 4;   // always 4x4 = 16 cells
    int cell_w = 0, cell_h = 0, comp_w = 0, comp_h = 0;
    int scale_x = 0, scale_y = 0;
    if (live_display) {
        // each cell = source_size / 4 (nearest decimation, factor 4)
        cell_w = common_sw / GRID_COLS;
        cell_h = common_sh / GRID_ROWS;
        scale_x = common_sw / cell_w;   // integer downscale factor in X
        scale_y = common_sh / cell_h;   // integer downscale factor in Y
        comp_w = GRID_COLS * cell_w;
        comp_h = GRID_ROWS * cell_h;

        // Producer: decimate all snapshots into the composite at 30 Hz and push.
        disp_producer = std::thread([&]() {
            // pre-allocate composite ONCE, pre-fill black (used cells overwritten each frame)
            std::vector<uint8_t> composite((size_t)comp_w * comp_h * 3, 0);
            using clk2 = std::chrono::steady_clock;
            const auto period = std::chrono::milliseconds(33);  // ~30 fps
            auto next = clk2::now() + period;
            // local scratch buffer per stream (avoid per-iter alloc)
            std::vector<std::vector<uint8_t>> snaps(streams.size());
            for (auto& sn : snaps) sn.assign((size_t)common_sw * common_sh * 3, 0);

            while (!disp_stop.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_until(next);
                next += period;
                for (size_t i = 0; i < streams.size(); ++i) {
                    Stream* s = streams[i].get();
                    // snapshot under lock (memcpy 1.2 MB; tens of microseconds)
                    {
                        std::lock_guard<std::mutex> lk(s->snap_mu);
                        if (s->snapshot.size() == snaps[i].size())
                            std::memcpy(snaps[i].data(), s->snapshot.data(), snaps[i].size());
                    }
                    int gx = (int)i % GRID_COLS, gy = (int)i / GRID_COLS;
                    if (gy >= GRID_ROWS) continue;
                    int dst_x0 = gx * cell_w, dst_y0 = gy * cell_h;
                    // 1/4 nearest decimation. Take every scale_x-th column and scale_y-th row.
                    for (int y = 0; y < cell_h; ++y) {
                        const uint8_t* src = snaps[i].data()
                                          + (size_t)(y * scale_y) * s->sw * 3;
                        uint8_t* dst = composite.data()
                                    + ((size_t)(dst_y0 + y) * comp_w + dst_x0) * 3;
                        for (int x = 0; x < cell_w; ++x) {
                            const uint8_t* p = src + (x * scale_x) * 3;
                            dst[0] = p[0]; dst[1] = p[1]; dst[2] = p[2];
                            dst += 3;
                        }
                    }
                }
                // Push a copy into the 1-slot leaky queue. Older frames are dropped automatically.
                disp_slot.push(std::vector<uint8_t>(composite));
            }
        });

        // Consumer: pop from leaky slot, write atomically to disp child, respawn on failure.
        disp_consumer = std::thread([&]() {
            Subprocess disp{};
            auto open_disp = [&]() {
                if (live_display == 1) disp = ffplay_display(comp_w, comp_h, 30.0, "yolo11n multi");
                else                    disp = ffmpeg_tcp_streamer(comp_w, comp_h, 30.0, 5000);
                std::fprintf(stderr,
                    "[display] composite pid=%d  %dx%d  4x4 grid  cell %dx%d  (scale 1/%dx1/%d)\n",
                    (int)disp.pid, comp_w, comp_h, cell_w, cell_h, scale_x, scale_y);
            };
            open_disp();
            std::vector<uint8_t> frame;
            while (!disp_stop.load(std::memory_order_relaxed) && disp_slot.pop(frame)) {
                if (disp.fd < 0) open_disp();
                if (!write_full(disp.fd, frame.data(), frame.size())) {
                    std::fprintf(stderr, "[display] viewer disconnected; respawning...\n");
                    close_sub(disp);
                }
            }
            close_sub(disp);
        });
    }

    // ---- Wait for all threads ----
    for (auto& t : decoders) t.join();
    raw_q.close();
    for (auto& t : preprocs) t.join();
    for (auto& t : workers) t.join();
    done_q.close();
    drawer.join();
    for (auto& sp : streams) sp->writer_thread.join();
    g_shutdown.store(true);
    if (stats_t.joinable()) stats_t.join();
    if (sig_watcher.joinable()) sig_watcher.join();

    auto t1 = clk::now();
    double total_s = std::chrono::duration<double>(t1 - t0).count();
    int inf_n = frames_inferred.load();
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

    // First, close all writer stdin pipes simultaneously so every ffmpeg child
    // gets EOF and can begin writing its mp4 trailer in parallel. Then wait up to
    // 15 s per child for clean exit; SIGTERM/SIGKILL only as fallback.
    for (auto& sp : streams) {
        if (sp->reader.fd >= 0) { ::close(sp->reader.fd); sp->reader.fd = -1; }
        if (sp->writer.fd >= 0) { ::close(sp->writer.fd); sp->writer.fd = -1; }
    }
    for (auto& sp : streams) {
        for (Subprocess* sub : { &sp->reader, &sp->writer }) {
            if (sub->pid <= 0) continue;
            int st = 0;
            bool done = false;
            for (int i = 0; i < 150; ++i) {            // up to 15 s
                pid_t r = waitpid(sub->pid, &st, WNOHANG);
                if (r == sub->pid) { done = true; break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (!done) ::kill(sub->pid, SIGTERM);
            for (int i = 0; i < 30 && !done; ++i) {
                pid_t r = waitpid(sub->pid, &st, WNOHANG);
                if (r == sub->pid) { done = true; break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (!done) { ::kill(sub->pid, SIGKILL); waitpid(sub->pid, &st, 0); }
            sub->pid = -1;
        }
    }
    if (live_display) {
        disp_stop.store(true);
        disp_slot.close();
        if (disp_producer.joinable()) disp_producer.join();
        if (disp_consumer.joinable()) disp_consumer.join();
    }
    axr_destroy(AXR_OBJECT(ctx));
    return 0;
}
