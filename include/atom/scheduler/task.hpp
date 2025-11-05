#pragma once

#include "../core/types.hpp"
#include "../core/tensor.hpp"
#include "../core/model_interface.hpp"
#include <functional>
#include <future>
#include <vector>
#include <set>

namespace atom::scheduler {

using TaskId = atom::core::u64;

// Task status
enum class TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled
};

// Task result
struct TaskResult {
    TaskId task_id;
    TaskStatus status;
    std::vector<atom::core::Tensor> outputs;
    atom::core::Duration execution_time;
    std::optional<atom::core::Error> error;
};

// Task definition
class Task {
public:
    using Callback = std::function<void(const TaskResult&)>;
    
    Task(TaskId id, 
         atom::core::ModelPtr model,
         std::vector<atom::core::Tensor> inputs,
         atom::core::Priority priority = atom::core::Priority::Normal);
    
    ~Task() = default;
    
    // Accessors
    TaskId GetId() const { return id_; }
    atom::core::Priority GetPriority() const { return priority_; }
    TaskStatus GetStatus() const { return status_; }
    const std::vector<atom::core::Tensor>& GetInputs() const { return inputs_; }
    atom::core::ModelPtr GetModel() const { return model_; }
    
    // Dependencies
    void AddDependency(TaskId dep_id);
    void RemoveDependency(TaskId dep_id);
    const std::set<TaskId>& GetDependencies() const { return dependencies_; }
    bool HasDependencies() const { return !dependencies_; }
    
    // Callbacks
    void SetCallback(Callback callback) { callback_ = std::move(callback); }
    void InvokeCallback(const TaskResult& result);
    
    // Execution
    void SetStatus(TaskStatus status) { status_ = status; }
    void SetStartTime(atom::core::TimePoint time) { start_time_ = time; }
    void SetEndTime(atom::core::TimePoint time) { end_time_ = time; }
    
    atom::core::Duration GetExecutionTime() const {
        if (start_time_ && end_time_) {
            return *end_time_ - *start_time_;
        }
        return atom::core::Duration::zero();
    }
    
    // Result management
    void SetResult(TaskResult result) { result_ = std::move(result); }
    const std::optional<TaskResult>& GetResult() const { return result_; }
    
    // Comparison for priority queue
    bool operator<(const Task& other) const {
        return priority_ < other.priority_;
    }
    
private:
    TaskId id_;
    atom::core::ModelPtr model_;
    std::vector<atom::core::Tensor> inputs_;
    atom::core::Priority priority_;
    TaskStatus status_{TaskStatus::Pending};
    
    std::set<TaskId> dependencies_;
    Callback callback_;
    
    std::optional<atom::core::TimePoint> start_time_;
    std::optional<atom::core::TimePoint> end_time_;
    std::optional<TaskResult> result_;
};

using TaskPtr = std::shared_ptr<Task>;

} // namespace atom::scheduler
