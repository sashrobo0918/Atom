#pragma once

#include <atom/core/model_interface.hpp>
#include <atom/inference/tensorrt_backend.hpp>

namespace atom::models {

class ResNet50 : public atom::core::ModelBase {
public:
    ResNet50() : ModelBase("ResNet50", "1.0.0") {
        metadata_.name = "ResNet50";
        metadata_.description = "ResNet50 image classification model";
        metadata_.input_names = {"input"};
        metadata_.output_names = {"output"};
        metadata_.input_shapes = {{1, 3, 224, 224}};
        metadata_.output_shapes = {{1, 1000}};
        metadata_.input_types = {atom::core::DataType::Float32};
        metadata_.output_types = {atom::core::DataType::Float32};
    }
    
    atom::core::Result<void> Initialize(const std::string& model_path, 
                                       const atom::core::InferenceOptions& options) override {
        backend_ = std::make_unique<atom::inference::TensorRTBackend>();
        
        auto init_result = backend_->Initialize(options.device);
        if (!init_result) return std::unexpected(init_result.error());
        
        auto load_result = backend_->LoadModel(model_path);
        if (!load_result) return std::unexpected(load_result.error());
        
        initialized_ = true;
        device_ = options.device;
        return {};
    }
    
    atom::core::Result<void> Warmup() override {
        if (!initialized_) {
            return std::unexpected(ATOM_ERROR(atom::core::ErrorCode::InvalidArgument,
                "Model not initialized"));
        }
        
        auto dummy_input = atom::core::Tensor::Create({1, 3, 224, 224}, 
            atom::core::DataType::Float32, device_);
        if (!dummy_input) return std::unexpected(dummy_input.error());
        
        std::vector<atom::core::Tensor> inputs = {*dummy_input};
        auto result = backend_->Execute(inputs);
        if (!result) return std::unexpected(result.error());
        
        return {};
    }
    
    void Shutdown() override {
        if (backend_) {
            backend_->Shutdown();
        }
        initialized_ = false;
    }
    
    atom::core::Result<std::vector<atom::core::Tensor>> Infer(
        const std::vector<atom::core::Tensor>& inputs) override {
        
        if (!ValidateInputs(inputs)) {
            return std::unexpected(ATOM_ERROR(atom::core::ErrorCode::InvalidArgument,
                "Invalid inputs"));
        }
        
        return backend_->Execute(inputs);
    }
    
    atom::core::Result<std::vector<atom::core::Tensor>> InferAsync(
        const std::vector<atom::core::Tensor>& inputs) override {
        return Infer(inputs);
    }
    
    atom::core::BackendType GetBackendType() const override {
        return atom::core::BackendType::TensorRT;
    }
    
    size_t GetMemoryUsage() const override {
        return 98 * 1024 * 1024; // ~98 MB for ResNet50
    }
    
private:
    std::unique_ptr<atom::inference::TensorRTBackend> backend_;
};

} // namespace atom::models
