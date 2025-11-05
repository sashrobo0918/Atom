#pragma once

#include "../core/types.hpp"
#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>
#include <atomic>

namespace atom::scheduler {

class ThreadPool {
public:
    using Task = std::function<void()>;
    
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency());
    ~ThreadPool();
    
    // Delete copy/move
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;
    
    // Task submission
    template<typename F, typename... Args>
    auto Submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
        using ReturnType = decltype(f(args...));
        
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        auto future = task->get_future();
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stopped_) {
                throw std::runtime_error("ThreadPool is stopped");
            }
            tasks_.emplace([task]() { (*task)(); });
        }
        
        condition_.notify_one();
        return future;
    }
    
    // Control
    void Stop();
    void WaitAll();
    
    // Query
    size_t GetThreadCount() const { return workers_.size(); }
    size_t GetQueuedTaskCount() const;
    size_t GetActiveTaskCount() const { return active_count_.load(); }
    
private:
    std::vector<std::thread> workers_;
    std::queue<Task> tasks_;
    
    mutable std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::condition_variable wait_condition_;
    
    std::atomic<bool> stopped_{false};
    std::atomic<size_t> active_count_{0};
    
    void WorkerThread();
};

} // namespace atom::scheduler
