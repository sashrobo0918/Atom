#pragma once

#include "../core/types.hpp"
#include "../core/tensor.hpp"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

namespace atom::data {

template<typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t max_size = 0) : max_size_(max_size) {}
    
    bool Push(T item, std::optional<atom::core::Duration> timeout = std::nullopt) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (max_size_ > 0) {
            if (timeout) {
                if (!not_full_.wait_for(lock, *timeout, [this]() {
                    return queue_.size() < max_size_ || stopped_;
                })) {
                    return false; // Timeout
                }
            } else {
                not_full_.wait(lock, [this]() {
                    return queue_.size() < max_size_ || stopped_;
                });
            }
        }
        
        if (stopped_) return false;
        
        queue_.push(std::move(item));
        not_empty_.notify_one();
        return true;
    }
    
    std::optional<T> Pop(std::optional<atom::core::Duration> timeout = std::nullopt) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (timeout) {
            if (!not_empty_.wait_for(lock, *timeout, [this]() {
                return !queue_.empty() || stopped_;
            })) {
                return std::nullopt; // Timeout
            }
        } else {
            not_empty_.wait(lock, [this]() {
                return !queue_.empty() || stopped_;
            });
        }
        
        if (stopped_ && queue_.empty()) {
            return std::nullopt;
        }
        
        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }
    
    size_t Size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
    bool Empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
    
    void Stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }
    
    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty()) {
            queue_.pop();
        }
    }
    
private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<T> queue_;
    size_t max_size_;
    bool stopped_{false};
};

} // namespace atom::data
