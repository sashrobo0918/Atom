#pragma once

#include "queue.hpp"
#include "preprocessor.hpp"
#include "../core/types.hpp"
#include <thread>
#include <atomic>
#include <functional>

namespace atom::data {

// Pipeline stage
template<typename InputT, typename OutputT>
class PipelineStage {
public:
    using TransformFunc = std::function<atom::core::Result<OutputT>(const InputT&)>;
    
    explicit PipelineStage(std::string name, TransformFunc transform, size_t num_workers = 1)
        : name_(std::move(name))
        , transform_(std::move(transform))
        , num_workers_(num_workers)
        , input_queue_(1000)
        , output_queue_(1000) {}
    
    void Start() {
        if (running_) return;
        running_ = true;
        
        for (size_t i = 0; i < num_workers_; ++i) {
            workers_.emplace_back([this]() { WorkerLoop(); });
        }
    }
    
    void Stop() {
        if (!running_) return;
        running_ = false;
        input_queue_.Stop();
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
    }
    
    bool Push(InputT input) {
        return input_queue_.Push(std::move(input));
    }
    
    std::optional<OutputT> Pop(std::optional<atom::core::Duration> timeout = std::nullopt) {
        return output_queue_.Pop(timeout);
    }
    
    const std::string& GetName() const { return name_; }
    size_t GetInputQueueSize() const { return input_queue_.Size(); }
    size_t GetOutputQueueSize() const { return output_queue_.Size(); }
    
private:
    std::string name_;
    TransformFunc transform_;
    size_t num_workers_;
    std::atomic<bool> running_{false};
    
    ThreadSafeQueue<InputT> input_queue_;
    ThreadSafeQueue<OutputT> output_queue_;
    std::vector<std::thread> workers_;
    
    void WorkerLoop() {
        while (running_) {
            auto input = input_queue_.Pop(std::chrono::milliseconds(100));
            if (!input) continue;
            
            auto result = transform_(*input);
            if (result) {
                output_queue_.Push(std::move(*result));
            }
        }
    }
};

// Complete data pipeline
class DataPipeline {
public:
    DataPipeline() = default;
    ~DataPipeline() { Stop(); }
    
    void Start();
    void Stop();
    bool IsRunning() const { return running_; }
    
private:
    std::atomic<bool> running_{false};
};

} // namespace atom::data
