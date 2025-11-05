#pragma once

#include "task.hpp"
#include <map>
#include <set>
#include <vector>
#include <mutex>

namespace atom::scheduler {

class DependencyGraph {
public:
    DependencyGraph() = default;
    ~DependencyGraph() = default;
    
    // Graph operations
    void AddTask(TaskPtr task);
    void RemoveTask(TaskId task_id);
    bool HasTask(TaskId task_id) const;
    
    void AddDependency(TaskId from_id, TaskId to_id);
    void RemoveDependency(TaskId from_id, TaskId to_id);
    
    // Query
    std::vector<TaskId> GetReadyTasks() const;
    std::vector<TaskId> GetDependents(TaskId task_id) const;
    bool HasCycle() const;
    
    // Completion
    void MarkCompleted(TaskId task_id);
    bool IsCompleted(TaskId task_id) const;
    
    // Topological sort
    atom::core::Result<std::vector<TaskId>> TopologicalSort() const;
    
    // Statistics
    size_t GetTaskCount() const;
    size_t GetPendingCount() const;
    size_t GetCompletedCount() const;
    
    void Clear();
    
private:
    mutable std::shared_mutex mutex_;
    
    std::map<TaskId, TaskPtr> tasks_;
    std::map<TaskId, std::set<TaskId>> adjacency_list_;  // task_id -> dependents
    std::map<TaskId, std::set<TaskId>> reverse_adjacency_; // task_id -> dependencies
    std::set<TaskId> completed_tasks_;
    
    bool HasCycleDFS(TaskId node, std::set<TaskId>& visited, 
                     std::set<TaskId>& rec_stack) const;
};

} // namespace atom::scheduler
