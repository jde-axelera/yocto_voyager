// concurrency.h
//
// Two tiny thread-safe primitives used across the pipeline:
//
//   BoundedQueue<T> — FIFO with a fixed capacity. push() blocks if full, pop()
//                     blocks if empty. close() unblocks both. Used between
//                     pipeline stages where every item must be processed.
//
//   LeakyOne<T>     — 1-slot leaky queue. push() always succeeds and overwrites
//                     any previous unread item (oldest dropped). pop() blocks
//                     until something is available. Used for the display path
//                     so the inference pipeline is never blocked by a slow
//                     viewer or network.
//
// Both are header-only templates.
#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <utility>

namespace yvm {

template <class T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t cap) : cap_(cap) {}

    /// Block until there's room, then move-push v. No-op if close()'d.
    void push(T&& v) {
        std::unique_lock<std::mutex> lk(mu_);
        not_full_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
        if (closed_) return;
        q_.push(std::move(v));
        not_empty_.notify_one();
    }

    /// Block until a value is available; return false once the queue is
    /// closed and empty (signal for consumers to exit).
    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(mu_);
        not_empty_.wait(lk, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop();
        not_full_.notify_one();
        return true;
    }

    /// Idempotent. Wakes all waiters; subsequent push() calls are no-ops.
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

template <class T>
class LeakyOne {
public:
    /// Always succeeds. Overwrites any previously-unread value. No-op if closed.
    void push(T&& v) {
        std::lock_guard<std::mutex> lk(mu_);
        if (closed_) return;
        slot_ = std::make_unique<T>(std::move(v));
        not_empty_.notify_one();
    }

    /// Block until a value is available; return false once closed and empty.
    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(mu_);
        not_empty_.wait(lk, [&] { return slot_ || closed_; });
        if (!slot_) return false;
        out = std::move(*slot_);
        slot_.reset();
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lk(mu_);
        closed_ = true;
        not_empty_.notify_all();
    }

private:
    std::unique_ptr<T> slot_;
    std::mutex mu_;
    std::condition_variable not_empty_;
    bool closed_ = false;
};

}  // namespace yvm
