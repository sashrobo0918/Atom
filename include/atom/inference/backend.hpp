#pragma once

#include "../core/types.hpp"
#include "../core/tensor.hpp"
#include <vector>
#include <memory>

namespace atom::inference {

// Base interface for inference backends
class IBackend {
public:
    virtual ~IBackend() = default;
    
    // Initialization
    virtual atom::core::Result<void> Initialize(const atom::core::DeviceInfo& device) = 0;
    virtual void Shutdown() = 0;
    
    // Model loading
    virtual atom::core::Result<void> LoadModel(const std::string& model_path) = 0;
    virtual void UnloadModel() = 0;
    
    // Inference
    virtual atom::core::Result<std::vector<atom::core::Tensor>> Execute(
        const std::vector<atom::core::Tensor>& inputs) = 0;
    
    // Query
    virtual atom::core::BackendType GetType() const = 0;
    virtual bool IsInitialized() const = 0;
    virtual bool IsModelLoaded() const = 0;
    
    // Optimization
    virtual atom::core::Result<void> OptimizeForBatchSize(size_t batch_size) = 0;
    virtual atom::core::Result<void> SetPrecision(atom::core::DataType precision) = 0;
};

using BackendPtr = std::shared_ptr<IBackend>;
using UniqueBackendPtr = std::unique_ptr<IBackend>;

} // namespace atom::inference
