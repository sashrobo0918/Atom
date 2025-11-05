#pragma once

#include "backend.hpp"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace atom::inference {

// Forward declaration of OptiCareTRT (assumed to be provided by TensorRT backend)
template<size_t NumOutputs>
class OptiCareTRT {
public:
    OptiCareTRT() = default;
    virtual ~OptiCareTRT() = default;
    
    virtual bool Initialize(const std::string& engine_path, int device_id) = 0;
    virtual bool Infer(void** inputs, void** outputs, cudaStream_t stream = nullptr) = 0;
    virtual void Destroy() = 0;
    
    virtual std::vector<nvinfer1::Dims> GetInputDims() const = 0;
    virtual std::vector<nvinfer1::Dims> GetOutputDims() const = 0;
    virtual size_t GetInputSize(int index) const = 0;
    virtual size_t GetOutputSize(int index) const = 0;
};

// TensorRT backend implementation
class TensorRTBackend : public IBackend {
public:
    TensorRTBackend();
    ~TensorRTBackend() override;
    
    // IBackend interface
    atom::core::Result<void> Initialize(const atom::core::DeviceInfo& device) override;
    void Shutdown() override;
    
    atom::core::Result<void> LoadModel(const std::string& model_path) override;
    void UnloadModel() override;
    
    atom::core::Result<std::vector<atom::core::Tensor>> Execute(
        const std::vector<atom::core::Tensor>& inputs) override;
    
    atom::core::BackendType GetType() const override { 
        return atom::core::BackendType::TensorRT; 
    }
    
    bool IsInitialized() const override { return initialized_; }
    bool IsModelLoaded() const override { return model_loaded_; }
    
    atom::core::Result<void> OptimizeForBatchSize(size_t batch_size) override;
    atom::core::Result<void> SetPrecision(atom::core::DataType precision) override;
    
    // TensorRT-specific methods
    atom::core::Result<void> BuildEngineFromONNX(const std::string& onnx_path);
    atom::core::Result<void> SaveEngine(const std::string& engine_path) const;
    
private:
    bool initialized_{false};
    bool model_loaded_{false};
    atom::core::DeviceInfo device_;
    std::string model_path_;
    
    // TensorRT objects
    std::shared_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    
    cudaStream_t stream_{nullptr};
    
    // Bindings
    std::vector<void*> device_bindings_;
    std::vector<size_t> binding_sizes_;
    
    // Helper methods
    atom::core::Result<void> CreateExecutionContext();
    atom::core::Result<void> AllocateBindings();
    void FreeBindings();
};

} // namespace atom::inference
