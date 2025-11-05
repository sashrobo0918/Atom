#pragma once

#include "model_interface.hpp"
#include "../logging/logger.hpp"
#include "../logging/metrics.hpp"
#include <atomic>

namespace atom::core {

// Wrapper that adds logging, metrics, and error handling to models
class ModelWrapper : public IModel {
public:
    explicit ModelWrapper(ModelPtr wrapped_model);
    ~ModelWrapper() override = default;
    
    // IModel interface implementation
    Result<void> Initialize(const std::string& model_path, const InferenceOptions& options) override;
    Result<void> Warmup() override;
    void Shutdown() override;
    
    Result<std::vector<Tensor>> Infer(const std::vector<Tensor>& inputs) override;
    Result<std::vector<Tensor>> InferAsync(const std::vector<Tensor>& inputs) override;
    
    ModelMetadata GetMetadata() const override;
    std::string GetName() const override;
    std::string GetVersion() const override;
    BackendType GetBackendType() const override;
    
    bool ValidateInputs(const std::vector<Tensor>& inputs) const override;
    bool IsInitialized() const override;
    
    size_t GetMemoryUsage() const override;
    DeviceInfo GetDevice() const override;
    
    // Statistics
    struct Statistics {
        std::atomic<uint64_t> inference_count{0};
        std::atomic<uint64_t> success_count{0};
        std::atomic<uint64_t> error_count{0};
        std::atomic<uint64_t> total_latency_ns{0};
        
        double GetAverageLatencyMs() const {
            auto count = inference_count.load();
            if (count == 0) return 0.0;
            return (total_latency_ns.load() / 1000000.0) / count;
        }
    };
    
    const Statistics& GetStatistics() const { return stats_; }
    void ResetStatistics();
    
private:
    ModelPtr wrapped_model_;
    Statistics stats_;
};

} // namespace atom::core
