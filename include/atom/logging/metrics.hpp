#pragma once

#include "../core/types.hpp"
#include <string>
#include <map>
#include <atomic>
#include <mutex>
#include <chrono>

namespace atom::logging {

// Metric types
class Counter {
public:
    Counter() = default;
    void Increment(atom::core::u64 value = 1) { value_ += value; }
    void Reset() { value_ = 0; }
    atom::core::u64 Get() const { return value_.load(); }
    
private:
    std::atomic<atom::core::u64> value_{0};
};

class Gauge {
public:
    Gauge() = default;
    void Set(atom::core::f64 value) { value_.store(value); }
    void Add(atom::core::f64 delta) { 
        atom::core::f64 current = value_.load();
        value_.store(current + delta);
    }
    atom::core::f64 Get() const { return value_.load(); }
    
private:
    std::atomic<atom::core::f64> value_{0.0};
};

class Histogram {
public:
    Histogram() = default;
    void Observe(atom::core::f64 value);
    atom::core::f64 GetMean() const;
    atom::core::f64 GetMin() const { return min_.load(); }
    atom::core::f64 GetMax() const { return max_.load(); }
    atom::core::u64 GetCount() const { return count_.load(); }
    void Reset();
    
private:
    std::atomic<atom::core::f64> sum_{0.0};
    std::atomic<atom::core::f64> min_{std::numeric_limits<atom::core::f64>::max()};
    std::atomic<atom::core::f64> max_{std::numeric_limits<atom::core::f64>::lowest()};
    std::atomic<atom::core::u64> count_{0};
};

// Metrics registry
class MetricsRegistry {
public:
    static MetricsRegistry& Instance();
    
    MetricsRegistry(const MetricsRegistry&) = delete;
    MetricsRegistry& operator=(const MetricsRegistry&) = delete;
    MetricsRegistry(MetricsRegistry&&) = delete;
    MetricsRegistry& operator=(MetricsRegistry&&) = delete;
    
    // Register metrics
    Counter& RegisterCounter(const std::string& name);
    Gauge& RegisterGauge(const std::string& name);
    Histogram& RegisterHistogram(const std::string& name);
    
    // Get metrics
    Counter* GetCounter(const std::string& name);
    Gauge* GetGauge(const std::string& name);
    Histogram* GetHistogram(const std::string& name);
    
    // Export
    std::string ExportJSON() const;
    std::string ExportPrometheus() const;
    
    void Clear();
    
private:
    MetricsRegistry() = default;
    ~MetricsRegistry() = default;
    
    mutable std::shared_mutex mutex_;
    std::map<std::string, Counter> counters_;
    std::map<std::string, Gauge> gauges_;
    std::map<std::string, Histogram> histograms_;
};

// RAII timer for measuring durations
class ScopedTimer {
public:
    explicit ScopedTimer(Histogram& histogram)
        : histogram_(histogram)
        , start_(std::chrono::high_resolution_clock::now()) {}
    
    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start_).count();
        histogram_.Observe(duration);
    }
    
private:
    Histogram& histogram_;
    std::chrono::high_resolution_clock::time_point start_;
};

} // namespace atom::logging
