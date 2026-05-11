// subprocess.h
//
// All ffmpeg/gst-launch child-process plumbing in one place. We never link
// libavcodec/libgstreamer directly; instead, we fork+exec the system binaries
// and stream raw frames through pipes. This keeps the cross-build minimal and
// lets the Rockchip MPP hardware codecs do video decode/encode for us.
#pragma once

#include <cstdint>
#include <string>
#include <sys/types.h>

namespace yvm {

/// A child process plus a pipe fd we use to talk to it.
///   reader-style: fd is the parent's READ end (child's stdout).
///   writer-style: fd is the parent's WRITE end (child's stdin).
struct Subprocess {
    pid_t pid = -1;
    int   fd  = -1;
};

/// Probe a video file with ffprobe; out-params are populated from its stream 0.
bool probe_video(const std::string& path, int& w, int& h, double& fps, int64_t& nframes);

/// Spawn ffmpeg as a paced raw-BGR reader.
///
///   ffmpeg -stream_loop -1 -re -i <path> -r <target_fps> -f rawvideo -pix_fmt bgr24 -
///
/// The looping input + `-re` pacing + `-r` re-clocking gives a steady output rate
/// regardless of how fast we drain the pipe.
Subprocess ffmpeg_reader(const std::string& path, int& w, int& h, double& fps,
                         int64_t& nframes, int target_fps);

/// Spawn ffmpeg as a V4L2 USB-camera raw-BGR reader.
///
///   ffmpeg -f v4l2 -input_format mjpeg -framerate <fps> -video_size <wxh>
///          -i /dev/videoN -f rawvideo -pix_fmt bgr24 -
///
/// MJPEG is used as the on-the-wire format so the higher resolutions / framerates
/// supported by most UVC webcams (e.g. Logitech C-series) are reachable. ffmpeg
/// decodes MJPEG and converts to BGR24 in software. There is no EOF for a USB
/// source — the pipe stays open until the parent closes the read end.
Subprocess ffmpeg_v4l2_reader(const std::string& device, int w, int h, int fps);

/// Spawn ffmpeg as a raw-BGR writer to an MP4 file using the Rockchip MPP H.264 encoder.
Subprocess ffmpeg_writer(const std::string& path, int w, int h, double fps);

/// Spawn gst-launch-1.0 to display raw BGR on the local X server (DISPLAY=:0).
/// Pipeline: fdsrc fd=0 ! rawvideoparse ! queue leaky=downstream ! videoconvert ! autovideosink sync=false
Subprocess gst_local_display(int w, int h, double fps, const char* title);

/// Spawn ffmpeg as a TCP MPEG-TS H.264 listener on the given port.
/// The first connecting client receives the H.264 stream; the encoder is h264_rkmpp.
Subprocess ffmpeg_tcp_streamer(int w, int h, double fps, int port);

/// Blocking byte-exact read from fd. Returns false on EOF / error / short read.
bool read_full(int fd, void* buf, size_t n);

/// Blocking byte-exact write to fd. Returns false on EPIPE / error / short write.
bool write_full(int fd, const void* buf, size_t n);

/// Close fd if open, then attempt clean SIGTERM-then-SIGKILL of the child:
///   1. Close the fd (signals EOF to the child).
///   2. Wait up to 2 s for clean exit (polled with waitpid WNOHANG).
///   3. SIGTERM, wait 1 s.
///   4. SIGKILL as final fallback.
void close_sub(Subprocess& s);

}  // namespace yvm
