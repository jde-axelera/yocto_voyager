// dma_heap.cpp
#include "dma_heap.h"

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <linux/types.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace yvm {

// ---- Linux dma-heap ABI (subset). We don't link any kernel header for these. ----
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

InputBufferPool::InputBufferPool(size_t buf_size, int count) {
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

InputBufferPool::~InputBufferPool() {
    for (auto& b : bufs_) {
        if (b.ptr) munmap(b.ptr, b.size);
        if (b.fd >= 0) ::close(b.fd);
    }
}

int InputBufferPool::acquire() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return !free_.empty(); });
    int i = free_.front(); free_.pop();
    return i;
}

void InputBufferPool::release(int idx) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        free_.push(idx);
    }
    cv_.notify_one();
}

void InputBufferPool::fill_all(uint8_t byte) {
    for (auto& b : bufs_) std::memset(b.ptr, byte, b.size);
}

void InputBufferPool::sync_start_write(int fd) {
    dma_buf_sync s{ DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE };
    ioctl(fd, DMA_BUF_IOCTL_SYNC, &s);
}

void InputBufferPool::sync_end_write(int fd) {
    dma_buf_sync s{ DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE };
    ioctl(fd, DMA_BUF_IOCTL_SYNC, &s);
}

}  // namespace yvm
