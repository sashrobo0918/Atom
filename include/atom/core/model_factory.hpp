#pragma once

#include "model_interface.hpp"
#include <functional>
#include <map>
#include <mutex>

namespace atom::core {

// Factory for creating models dynamically
class ModelFactory {
public:
    using CreatorFunc = std::function<UniqueModelPtr()>;
    
    static ModelFactory& Instance();
    
    // Delete copy/move
    ModelFactory(const ModelFactory&) = delete;
    ModelFactory& operator=(const ModelFactory&) = delete;
    ModelFactory(ModelFactory&&) = delete;
    ModelFactory& operator=(ModelFactory&&) = delete;
    
    // Registration
    bool Register(const std::string& model_type, CreatorFunc creator);
    bool Unregister(const std::string& model_type);
    bool IsRegistered(const std::string& model_type) const;
    
    // Creation
    Result<UniqueModelPtr> Create(const std::string& model_type) const;
    
    // Query
    std::vector<std::string> GetRegisteredTypes() const;
    size_t GetRegisteredCount() const;
    
private:
    ModelFactory() = default;
    ~ModelFactory() = default;
    
    mutable std::shared_mutex mutex_;
    std::map<std::string, CreatorFunc> creators_;
};

// Helper class for automatic registration
template<typename T>
class ModelRegistrar {
public:
    explicit ModelRegistrar(const std::string& type_name) {
        ModelFactory::Instance().Register(type_name, []() -> UniqueModelPtr {
            return std::make_unique<T>();
        });
    }
};

// Macro for easy model registration
#define REGISTER_MODEL(ModelClass, TypeName) \
    namespace { \
        static atom::core::ModelRegistrar<ModelClass> \
            registrar_##ModelClass(TypeName); \
    }

} // namespace atom::core
