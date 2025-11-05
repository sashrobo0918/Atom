#include "atom/core/model_factory.hpp"

namespace atom::core {

ModelFactory& ModelFactory::Instance() {
    static ModelFactory instance;
    return instance;
}

bool ModelFactory::Register(const std::string& model_type, CreatorFunc creator) {
    std::unique_lock lock(mutex_);
    
    if (creators_.find(model_type) != creators_.end()) {
        return false; // Already registered
    }
    
    creators_[model_type] = std::move(creator);
    return true;
}

bool ModelFactory::Unregister(const std::string& model_type) {
    std::unique_lock lock(mutex_);
    return creators_.erase(model_type) > 0;
}

bool ModelFactory::IsRegistered(const std::string& model_type) const {
    std::shared_lock lock(mutex_);
    return creators_.find(model_type) != creators_.end();
}

Result<UniqueModelPtr> ModelFactory::Create(const std::string& model_type) const {
    std::shared_lock lock(mutex_);
    
    auto it = creators_.find(model_type);
    if (it == creators_.end()) {
        return std::unexpected(ATOM_ERROR(ErrorCode::ModelNotFound, 
            "Model type not registered: " + model_type));
    }
    
    try {
        return it->second();
    } catch (const std::exception& e) {
        return std::unexpected(ATOM_ERROR(ErrorCode::Unknown, 
            "Failed to create model: " + std::string(e.what())));
    }
}

std::vector<std::string> ModelFactory::GetRegisteredTypes() const {
    std::shared_lock lock(mutex_);
    
    std::vector<std::string> types;
    types.reserve(creators_.size());
    
    for (const auto& [type, _] : creators_) {
        types.push_back(type);
    }
    
    return types;
}

size_t ModelFactory::GetRegisteredCount() const {
    std::shared_lock lock(mutex_);
    return creators_.size();
}

} // namespace atom::core
