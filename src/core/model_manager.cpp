#include "atom/core/model_manager.hpp"
#include <chrono>

namespace atom::core {

ModelManager& ModelManager::Instance() {
    static ModelManager instance;
    return instance;
}

ModelManager::~ModelManager() {
    UnloadAll();
}

Result<void> ModelManager::LoadModel(
    const std::string& model_id,
    const std::string& model_type,
    const std::string& model_path,
    const InferenceOptions& options) {
    
    std::unique_lock lock(mutex_);
    
    if (models_.find(model_id) != models_.end()) {
        return std::unexpected(ATOM_ERROR(ErrorCode::InvalidArgument, 
            "Model already loaded: " + model_id));
    }
    
    auto model_result = ModelFactory::Instance().Create(model_type);
    if (!model_result) {
        return std::unexpected(model_result.error());
    }
    
    auto model = std::make_shared<IModel>(std::move(*model_result));
    auto init_result = (*model_result)->Initialize(model_path, options);
    if (!init_result) {
        return std::unexpected(init_result.error());
    }
    
    ModelEntry entry{
        .model = std::move(model),
        .model_type = model_type,
        .model_path = model_path,
        .options = options,
        .load_time = std::chrono::high_resolution_clock::now()
    };
    
    models_[model_id] = std::move(entry);
    return {};
}

Result<void> ModelManager::UnloadModel(const std::string& model_id) {
    std::unique_lock lock(mutex_);
    
    auto it = models_.find(model_id);
    if (it == models_.end()) {
        return std::unexpected(ATOM_ERROR(ErrorCode::ModelNotFound, 
            "Model not found: " + model_id));
    }
    
    it->second.model->Shutdown();
    models_.erase(it);
    return {};
}

Result<ModelPtr> ModelManager::GetModel(const std::string& model_id) const {
    std::shared_lock lock(mutex_);
    
    auto it = models_.find(model_id);
    if (it == models_.end()) {
        return std::unexpected(ATOM_ERROR(ErrorCode::ModelNotFound, 
            "Model not found: " + model_id));
    }
    
    return it->second.model;
}

bool ModelManager::HasModel(const std::string& model_id) const {
    std::shared_lock lock(mutex_);
    return models_.find(model_id) != models_.end();
}

std::vector<std::string> ModelManager::GetLoadedModels() const {
    std::shared_lock lock(mutex_);
    
    std::vector<std::string> ids;
    ids.reserve(models_.size());
    
    for (const auto& [id, _] : models_) {
        ids.push_back(id);
    }
    
    return ids;
}

Result<void> ModelManager::UnloadAll() {
    std::unique_lock lock(mutex_);
    
    for (auto& [_, entry] : models_) {
        entry.model->Shutdown();
    }
    
    models_.clear();
    return {};
}

size_t ModelManager::GetTotalMemoryUsage() const {
    std::shared_lock lock(mutex_);
    
    size_t total = 0;
    for (const auto& [_, entry] : models_) {
        total += entry.model->GetMemoryUsage();
    }
    
    return total;
}

} // namespace atom::core
