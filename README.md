# Atom AI Inference Framework

A high-performance, modular C++20 framework for AI model inference with advanced scheduling, data processing, and visualization capabilities.

## Features

- **Dynamic Model Registration**: Easily add new AI models through factory pattern
- **Advanced Scheduler**: Parallel task execution with dependency management
- **Multiple Backends**: TensorRT, ONNX, and CPU backends
- **Data Pipeline**: Preprocessing with OpenCV integration
- **Logging & Metrics**: Comprehensive logging and performance metrics
- **Profiling & Visualization**: Built-in profiling and dashboard
- **Exception Handling**: Robust error handling with C++20 std::expected
- **CUDA Integration**: Full CUDA and TensorRT support

## Architecture

```
Atom/
├── include/atom/           # Public headers
│   ├── core/              # Core types, tensors, model management
│   ├── inference/         # Backend implementations
│   ├── scheduler/         # Task scheduling and thread pool
│   ├── logging/           # Logger and metrics
│   ├── data/              # Data pipeline and preprocessing
│   └── viz/               # Profiling and visualization
├── src/                   # Implementation files
├── models/                # Model implementations
│   ├── detection/         # Object detection models (YOLO, etc.)
│   ├── classification/    # Classification models (ResNet, etc.)
│   ├── segmentation/      # Segmentation models
│   └── nlp/               # NLP models
└── examples/              # Example applications
```

## Requirements

- C++20 compatible compiler (GCC 11+, Clang 13+)
- CUDA 11.0+
- TensorRT 8.0+
- OpenCV 4.x
- Meson build system

## Building

```bash
cd atom
meson setup build
meson compile -C build
```

## Adding a New Model

1. Create model class inheriting from `ModelBase`:

```cpp
#include <atom/core/model_interface.hpp>

class MyModel : public atom::core::ModelBase {
public:
    MyModel() : ModelBase("MyModel", "1.0.0") {
        // Initialize metadata
    }
    
    Result<void> Initialize(const std::string& path, 
                          const InferenceOptions& opts) override {
        // Load model
    }
    
    Result<std::vector<Tensor>> Infer(
        const std::vector<Tensor>& inputs) override {
        // Run inference
    }
    
    // Implement other virtual methods
};
```

2. Register the model:

```cpp
#include <atom/core/model_factory.hpp>

REGISTER_MODEL(MyModel, "my_model");
```

## Usage Example

```cpp
#include <atom/core/model_manager.hpp>
#include <atom/scheduler/scheduler.hpp>

int main() {
    // Create scheduler
    atom::scheduler::Scheduler sched;
    sched.Start();
    
    // Load model
    auto& mgr = atom::core::ModelManager::Instance();
    mgr.LoadModel("model1", "yolov8", "/path/to/model.engine");
    
    // Get model and submit task
    auto model = mgr.GetModel("model1");
    auto input = atom::core::Tensor::Create({1, 3, 640, 640}, 
                                           atom::core::DataType::Float32);
    
    auto task_id = sched.SubmitTask(*model, {*input});
    auto result = sched.WaitForTask(*task_id);
    
    // Cleanup
    sched.Stop();
    mgr.UnloadAll();
}
```

## Key Design Patterns

- **Factory Pattern**: Dynamic model and backend registration
- **Singleton Pattern**: Global access to managers and configuration
- **RAII**: Automatic resource management
- **Observer Pattern**: Task callbacks for asynchronous operations
- **Strategy Pattern**: Pluggable backends

## Performance

- Zero-copy tensor operations where possible
- Lock-free queues for high-throughput data pipeline
- CUDA streams for async operations
- Thread pool for parallel execution
- Memory pooling to reduce allocations

## License

[Your License Here]

## Contributing

1. Fork the repository
2. Create your feature branch
3. Implement your model/feature
4. Add tests
5. Submit pull request

## Contact

[Your Contact Information]
