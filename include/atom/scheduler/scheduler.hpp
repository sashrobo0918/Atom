#pragma once

#include "task.hpp"
#include "thread_pool.hpp"
#include "dependency_graph.hpp"
#include "../core/types.hpp"
#include <queue>
#include <map>
#include <atomic>

namespace atom::scheduler {

// Scheduler configuration
struct SchedulerConfig {
    size_t num_threads{std::thread::hardware_concurrency()};
    size_t max_queue_size{1000};
    bool enable_profiling{false};
    atom::core::Duration task_timeout{std::chrono::seconds(30)};
};

// Scheduler for parallel task execution
class Scheduler {
public:
    explicit Scheduler(const SchedulerConfig& config = SchedulerConfig{});
    ~Scheduler();
    
    // Delete copy/move
    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;
    Scheduler(Scheduler&&) = delete;
    Scheduler& operator=(Scheduler&&) = delete;
    
    // Lifecycle
    atom::core::Result<void> Start();
    void Stop();
    bool IsRunning() const { return running_; }
    
    // Task submission
    atom::core::Result<TaskId> SubmitTask(
        atom::core::ModelPtr model,
        std::vector<atom::core::Tensor> inputs,
        atom::core::Priority priority = atom::core::Priority::Normal,
        Task::Callback callback = nullptr
    );
    
    atom::core::Result<TaskId> SubmitTaskWithDependencies(
        atom::core::ModelPtr model,
        std::vector<atom::core::Tensor> inputs,
        std::vector<TaskId> dependencies,
        atom::core::Priority priority = atom::core::Priority::Normal,
        Task::Callback callback = nullptr
    );
    
    // Batch submission
    atom::core::Result<std::vector<TaskId>> SubmitBatch(
        const std::vector<std::pair<atom::core::ModelPtr, std::vector<atom::core::Tensor>>>& batch,
        atom::core::Priority priority = atom::core::Priority::Normal
    );
    
    // Task control
    atom::core::Result<void> CancelTask(TaskId task_id);
    atom::core::Result<TaskResult> WaitForTask(TaskId task_id, 
        std::optional<atom::core::Duration> timeout = std::nullopt);
    atom::core::Result<std::vector<TaskResult>> WaitForAll(
        const std::vector<TaskId>& task_ids,
        std::optional<atom::core::Duration> timeout = std::nullopt);
    
    // Query
    std::optional<TaskStatus> GetTaskStatus(TaskId task_id) const;
    size_t GetQueuedTaskCount() const;
    size_t GetRunningTaskCount() const;
    size_t GetCompletedTaskCount() const;
    
    // Statistics
    struct Statistics {
        std::atomic<uint64_t> total_tasks{0};
        std::atomic<uint64_t> completed_tasks{0};
        std::atomic<uint64_t> failed_tasks{0};
        std::atomic<uint64_t> cancelled_tasks{0};
        std::atomic<uint64_t> total_execution_time_ns{0};
        
        double GetAverageExecutionTimeMs() const {
            auto count = completed_tasks.load();
            if (count == 0) return 0.0;
            return (total_execution_time_ns.load() / 1000000.0) / count;
        }
    };
    
    const Statistics& GetStatistics() const { return stats_; }
    void ResetStatistics();
    
private:
    SchedulerConfig config_;
    std::unique_ptr<ThreadPool> thread_pool_;
    DependencyGraph dependency_graph_;
    
    std::atomic<bool> running_{false};
    std::atomic<TaskId> next_task_id_{1};
    
    mutable std::shared_mutex mutex_;
    std::map<TaskId, TaskPtr> all_tasks_;
    std::map<TaskId, std::promise<TaskResult>> task_promises_;
    
    // Priority queue for ready tasks
    using PriorityQueue = std::priority_queue<TaskPtr, std::vector<TaskPtr>, 
        std::function<bool(const TaskPtr&, const TaskPtr&)>>;
    PriorityQueue ready_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    Statistics stats_;
    
    // Worker thread
    std::thread scheduler_thread_;
    void SchedulerLoop();
    
    // Task execution
    void ExecuteTask(TaskPtr task);
    void OnTaskCompleted(TaskPtr task, const TaskResult& result);
    void OnTaskFailed(TaskPtr task, const atom::core::Error& error);
    
    // Helper methods
    TaskId GenerateTaskId() { return next_task_id_++; }
    void EnqueueReadyTasks();
};

} // namespace atom::scheduler
