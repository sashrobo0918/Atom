#pragma once

#include "../core/types.hpp"
#include <string>
#include <map>
#include <vector>
#include <chrono>
#include <mutex>

namespace atom::viz {

struct ProfileEntry {
    std::string name;
    atom::core::TimePoint start_time;
    atom::core::TimePoint end_time;
    atom::core::Duration duration;
    std::map<std::string, std::string> metadata;
};

class Profiler {
public:
    static Profiler& Instance();
    
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;
    Profiler(Profiler&&) = delete;
    Profiler& operator=(Profiler&&) = delete;
    
    void Enable() { enabled_ = true; }
    void Disable() { enabled_ = false; }
    bool IsEnabled() const { return enabled_; }
    
    void BeginSection(const std::string& name);
    void EndSection(const std::string& name);
    
    // Get profiling results
    std::vector<ProfileEntry> GetEntries() const;
    void Clear();
    
    // Export
    std::string ExportJSON() const;
    std::string ExportChrome Trace() const; // Chrome tracing format
    
private:
    Profiler() = default;
    ~Profiler() = default;
    
    bool enabled_{false};
    mutable std::mutex mutex_;
    std::vector<ProfileEntry> entries_;
    std::map<std::string, atom::core::TimePoint> active_sections_;
};

// RAII profile scope
class ProfileScope {
public:
    explicit ProfileScope(std::string name) : name_(std::move(name)) {
        if (Profiler::Instance().IsEnabled()) {
            Profiler::Instance().BeginSection(name_);
        }
    }
    
    ~ProfileScope() {
        if (Profiler::Instance().IsEnabled()) {
            Profiler::Instance().EndSection(name_);
        }
    }
    
private:
    std::string name_;
};

#define PROFILE_SCOPE(name) atom::viz::ProfileScope profile_scope_##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)

} // namespace atom::viz
