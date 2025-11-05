#pragma once

#include "types.hpp"
#include "tensor.hpp"
#include <vector>
#include <string>
#include <memory>

namespace atom::core {

// Base interface for all AI models
class IModel {
public:
    virtual ~IModel() = default;
    
    // Model lifecycle
    virtual Result<void> Initialize(const std::string& model_path, const InferenceOptions& options) = 0;
    virtual Result<void> Warmup() = 0;
    virtual void Shutdown() = 0;
    
    // Inference
    virtual Result<std::vector<Tensor>> Infer(const std::vector<Tensor>& inputs) = 0;
    virtual Result<std::vector<Tensor>> InferAsync(const std::vector<Tensor>& inputs) = 0;
    
    // Metadata
    virtual ModelMetadata GetMetadata() const = 0;
    virtual std::string GetName() const = 0;
    virtual std::string GetVersion() const = 0;
    virtual BackendType GetBackendType() const = 0;
    
    // Validation
    virtual bool ValidateInputs(const std::vector<Tensor>& inputs) const = 0;
    virtual bool IsInitialized() const = 0;
    
    // Resource management
    virtual size_t GetMemoryUsage() const = 0;
    virtual DeviceInfo GetDevice() const = 0;
};

// Abstract base class with common functionality
class ModelBase : public IModel {
public:
    explicit ModelBase(std::string name, std::string version = "1.0.0")
        : name_(std::move(name)), version_(std::move(version)), initialized_(false) {}
    
    ~ModelBase() override = default;
    
    std::string GetName() const override { return name_; }
    std::string GetVersion() const override { return version_; }
    bool IsInitialized() const override { return initialized_; }
    DeviceInfo GetDevice() const override { return device_; }
    
    ModelMetadata GetMetadata() const override { return metadata_; }
    
    bool ValidateInputs(const std::vector<Tensor>& inputs) const override {
        const auto& expected_shapes = metadata_.input_shapes;
        const auto& expected_types = metadata_.input_types;
        
        if (inputs.size() != expected_shapes.size()) {
            return false;
        }
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i].GetDataType() != expected_types[i]) {
                return false;
            }
            // Allow dynamic batch size (first dimension can vary)
            const auto& input_shape = inputs[i].GetShape();
            const auto& expected_shape = expected_shapes[i];
            if (input_shape.size() != expected_shape.size()) {
                return false;
            }
            for (size_t j = 1; j < input_shape.size(); ++j) {
                if (input_shape[j] != expected_shape[j]) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
protected:
    std::string name_;
    std::string version_;
    bool initialized_;
    DeviceInfo device_{DeviceType::CUDA, 0};
    ModelMetadata metadata_;
};

// Smart pointer type for models
using ModelPtr = std::shared_ptr<IModel>;
using UniqueModelPtr = std::unique_ptr<IModel>;

} // namespace atom::core
