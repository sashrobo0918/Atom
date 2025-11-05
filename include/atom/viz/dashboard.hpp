#pragma once

#include "../core/types.hpp"
#include "../logging/metrics.hpp"
#include <string>
#include <map>

namespace atom::viz {

// Dashboard for monitoring system status
class Dashboard {
public:
    static Dashboard& Instance();
    
    Dashboard(const Dashboard&) = delete;
    Dashboard& operator=(const Dashboard&) = delete;
    Dashboard(Dashboard&&) = delete;
    Dashboard& operator=(Dashboard&&) = delete;
    
    // Update dashboard data
    void UpdateModelStats(const std::string& model_id, const std::map<std::string, double>& stats);
    void UpdateSchedulerStats(const std::map<std::string, double>& stats);
    void UpdateSystemStats(const std::map<std::string, double>& stats);
    
    // Export dashboard data
    std::string ExportHTML() const;
    std::string ExportJSON() const;
    
    // HTTP server (optional, for web-based dashboard)
    atom::core::Result<void> StartServer(uint16_t port = 8080);
    void StopServer();
    
private:
    Dashboard() = default;
    ~Dashboard();
    
    mutable std::shared_mutex mutex_;
    std::map<std::string, std::map<std::string, double>> model_stats_;
    std::map<std::string, double> scheduler_stats_;
    std::map<std::string, double> system_stats_;
    
    bool server_running_{false};
    // HTTP server implementation would go here
};

} // namespace atom::viz
