#pragma once

#include "types.hpp"
#include <map>
#include <any>
#include <mutex>
#include <filesystem>

namespace atom::core {

class Config {
public:
    static Config& Instance();
    
    // Delete copy/move constructors
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;
    Config(Config&&) = delete;
    Config& operator=(Config&&) = delete;
    
    // Configuration methods
    void Set(const std::string& key, std::any value);
    
    template<typename T>
    std::optional<T> Get(const std::string& key) const {
        std::shared_lock lock(mutex_);
        auto it = config_map_.find(key);
        if (it == config_map_.end()) {
            return std::nullopt;
        }
        try {
            return std::any_cast<T>(it->second);
        } catch (const std::bad_any_cast&) {
            return std::nullopt;
        }
    }
    
    template<typename T>
    T GetOr(const std::string& key, T default_value) const {
        return Get<T>(key).value_or(std::move(default_value));
    }
    
    bool Has(const std::string& key) const;
    void Remove(const std::string& key);
    void Clear();
    
    // Load from file (JSON/YAML format)
    Result<void> LoadFromFile(const std::filesystem::path& path);
    Result<void> SaveToFile(const std::filesystem::path& path) const;
    
    // Predefined configuration keys
    static constexpr const char* KEY_NUM_THREADS = "scheduler.num_threads";
    static constexpr const char* KEY_MAX_BATCH_SIZE = "inference.max_batch_size";
    static constexpr const char* KEY_ENABLE_PROFILING = "profiling.enabled";
    static constexpr const char* KEY_LOG_LEVEL = "logging.level";
    static constexpr const char* KEY_CUDA_DEVICE = "cuda.device_id";
    
private:
    Config() = default;
    ~Config() = default;
    
    mutable std::shared_mutex mutex_;
    std::map<std::string, std::any> config_map_;
};

} // namespace atom::core
