#include "atom/core/config.hpp"
#include <fstream>

namespace atom::core {

Config& Config::Instance() {
    static Config instance;
    return instance;
}

void Config::Set(const std::string& key, std::any value) {
    std::unique_lock lock(mutex_);
    config_map_[key] = std::move(value);
}

bool Config::Has(const std::string& key) const {
    std::shared_lock lock(mutex_);
    return config_map_.find(key) != config_map_.end();
}

void Config::Remove(const std::string& key) {
    std::unique_lock lock(mutex_);
    config_map_.erase(key);
}

void Config::Clear() {
    std::unique_lock lock(mutex_);
    config_map_.clear();
}

Result<void> Config::LoadFromFile(const std::filesystem::path& path) {
    // Basic implementation - could be enhanced with JSON/YAML parsing
    if (!std::filesystem::exists(path)) {
        return std::unexpected(ATOM_ERROR(ErrorCode::InvalidArgument, 
            "Config file does not exist: " + path.string()));
    }
    
    // Placeholder for actual file loading
    return {};
}

Result<void> Config::SaveToFile(const std::filesystem::path& path) const {
    // Placeholder for actual file saving
    return {};
}

} // namespace atom::core
