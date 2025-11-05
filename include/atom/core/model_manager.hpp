#pragma once

#include "model_interface.hpp"
#include "model_factory.hpp"
#include <map>
#include <string>
#include <mutex>

namespace atom::core {

// Manages model instances
class ModelManager {
public:
    static ModelManager& Instance();
    
    // Delete copy/move
    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;
    ModelManager(ModelManager&&) = delete;
    ModelManager& operator=(ModelManager&&) = delete;
    
    // Model lifecycle
    Result<void> LoadModel(
        const std::string& model_id,
        const std::string& model_type,
        const std::string& model_path,
        const InferenceOptions& options = InferenceOptions{}
    );
    
    Result<void> UnloadModel(const std::string& model_id);
    Result<void> ReloadModel(const std::string& model_id);
    
    // Model access
    Result<ModelPtr> GetModel(const std::string& model_id) const;
    bool HasModel(const std::string& model_id) const;
    
    // Query
    std::vector<std::string> GetLoadedModels() const;
    size_t GetModelCount() const;
    
    // Batch operations
    Result<void> UnloadAll();
    Result<void> WarmupAll();
    
    // Memory management
    size_t GetTotalMemoryUsage() const;
    
private:
    ModelManager() = default;
    ~ModelManager();
    
    struct ModelEntry {
        ModelPtr model;
        std::string model_type;
        std::string model_path;
        InferenceOptions options;
        TimePoint load_time;
    };
    
    mutable std::shared_mutex mutex_;
    std::map<std::string, ModelEntry> models_;
};

} // namespace atom::core
