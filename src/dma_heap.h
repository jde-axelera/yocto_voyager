// dma_heap.h
//
// Tiny pool of dma-heap-backed buffers, used for the AIPU input dmabuf path.
//
// Voyager Linux doesn't ship linux/dma-heap.h, so we redeclare the ABI here.
// We use the `system` heap (cached system memory exposed as a dma-buf fd).
//
// Each buffer is allocated once, mmap'd CPU-side, and recycled through a
// blocking acquire()/release() cycle. The CPU writes (preproc) and AIPU reads
// (via the fd) need explicit cache sync calls — see sync_start_write() /
// sync_end_write().
#pragma once

#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <queue>
#include <vector>

namespace yvm {

class InputBufferPool {
public:
    struct Buf {
        int    fd  = -1;
        void*  ptr = nullptr;
        size_t size = 0;
    };

    /// Allocate `count` buffers of `buf_size` bytes each from /dev/dma_heap/system.
    /// Throws std::exit(1) on failure.
    InputBufferPool(size_t buf_size, int count);
    ~InputBufferPool();
    InputBufferPool(const InputBufferPool&) = delete;
    InputBufferPool& operator=(const InputBufferPool&) = delete;

    /// Block until a buffer is free, return its index.
    int  acquire();

    /// Return a previously-acquired buffer to the pool.
    void release(int idx);

    /// Random access to the buffer descriptors (read-only).
    const Buf& operator[](int i) const { return bufs_[i]; }

    /// memset every buffer to `byte`. Useful for pre-filling the AIPU padding
    /// regions with the model's int8 zero_point so we only ever have to write
    /// the *active* pixel region each frame.
    void fill_all(uint8_t byte);

    /// CPU→device cache sync: call before/after a CPU write that the AIPU
    /// will subsequently read. Wraps DMA_BUF_IOCTL_SYNC.
    static void sync_start_write(int fd);
    static void sync_end_write  (int fd);

    /// device→CPU cache sync: call before/after a CPU read of data the AIPU
    /// just wrote. Used for output dmabufs.
    static void sync_start_read (int fd);
    static void sync_end_read   (int fd);

private:
    std::vector<Buf>         bufs_;
    std::queue<int>          free_;
    std::mutex               mu_;
    std::condition_variable  cv_;
};

}  // namespace yvm
