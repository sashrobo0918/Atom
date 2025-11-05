#pragma once

#include <atom/core/model_interface.hpp>
#include <atom/inference/tensorrt_backend.hpp>

namespace atom::models {

class YOLOv8 : public atom::core::ModelBase {
public:
    YOLOv8() : ModelBase("YOLOv8", "1.0.0") {
        // Initialize metadata
        metadata_.name = "YOLOv8";
        metadata_.description = "YOLOv8 object detection model";
        metadata_.input_names = {"images"};
        metadata_.output_names = {"boxes", "scores", "classes"};
        metadata_.input_shapes = {{1, 3, 640, 640}};
        metadata_.output_shapes = {{1, 8400, 4}, {1, 8400, 80}, {1, 8400, 1}};
        metadata_.input_types = {atom::core::DataType::Float32};
        metadata_.output_types = {atom::core::DataType::Float32, 
                                  atom::core::DataType::Float32,
                                  atom::core::DataType::Float32};
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
        
        // Create dummy input
        auto dummy_input = atom::core::Tensor::Create({1, 3, 640, 640}, 
            atom::core::DataType::Float32, device_);
        if (!dummy_input) return std::unexpected(dummy_input.error());
        
        // Run inference once
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
        // For now, just call synchronous version
        return Infer(inputs);
    }
    
    atom::core::BackendType GetBackendType() const override {
        return atom::core::BackendType::TensorRT;
    }
    
    size_t GetMemoryUsage() const override {
        // Simplified - should query actual memory usage
        return 100 * 1024 * 1024; // 100 MB
    }
    
private:
    std::unique_ptr<atom::inference::TensorRTBackend> backend_;
};

} // namespace atom::models
