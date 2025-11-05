#pragma once

#include "backend.hpp"
#include <functional>
#include <map>
#include <mutex>

namespace atom::inference {

class BackendFactory {
public:
    using CreatorFunc = std::function<UniqueBackendPtr()>;
    
    static BackendFactory& Instance();
    
    BackendFactory(const BackendFactory&) = delete;
    BackendFactory& operator=(const BackendFactory&) = delete;
    BackendFactory(BackendFactory&&) = delete;
    BackendFactory& operator=(BackendFactory&&) = delete;
    
    // Registration
    bool Register(atom::core::BackendType type, CreatorFunc creator);
    bool Unregister(atom::core::BackendType type);
    
    // Creation
    atom::core::Result<UniqueBackendPtr> Create(atom::core::BackendType type) const;
    
    // Query
    bool IsAvailable(atom::core::BackendType type) const;
    std::vector<atom::core::BackendType> GetAvailableBackends() const;
    
private:
    BackendFactory() = default;
    ~BackendFactory() = default;
    
    mutable std::shared_mutex mutex_;
    std::map<atom::core::BackendType, CreatorFunc> creators_;
};

// Auto-registration helper
template<typename T>
class BackendRegistrar {
public:
    explicit BackendRegistrar(atom::core::BackendType type) {
        BackendFactory::Instance().Register(type, []() -> UniqueBackendPtr {
            return std::make_unique<T>();
        });
    }
};

#define REGISTER_BACKEND(BackendClass, BackendType) \
    namespace { \
        static atom::inference::BackendRegistrar<BackendClass> \
            registrar_##BackendClass(BackendType); \
    }

} // namespace atom::inference
