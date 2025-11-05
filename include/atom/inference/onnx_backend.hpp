#pragma once

#include "backend.hpp"
#include <NvOnnxParser.h>

namespace atom::inference {

class ONNXBackend : public IBackend {
public:
    ONNXBackend();
    ~ONNXBackend() override;
    
    atom::core::Result<void> Initialize(const atom::core::DeviceInfo& device) override;
    void Shutdown() override;
    
    atom::core::Result<void> LoadModel(const std::string& model_path) override;
    void UnloadModel() override;
    
    atom::core::Result<std::vector<atom::core::Tensor>> Execute(
        const std::vector<atom::core::Tensor>& inputs) override;
    
    atom::core::BackendType GetType() const override { 
        return atom::core::BackendType::ONNX; 
    }
    
    bool IsInitialized() const override { return initialized_; }
    bool IsModelLoaded() const override { return model_loaded_; }
    
    atom::core::Result<void> OptimizeForBatchSize(size_t batch_size) override;
    atom::core::Result<void> SetPrecision(atom::core::DataType precision) override;
    
private:
    bool initialized_{false};
    bool model_loaded_{false};
    atom::core::DeviceInfo device_;
    
    // Will internally use TensorRT for execution
    std::unique_ptr<TensorRTBackend> trt_backend_;
};

} // namespace atom::inference
