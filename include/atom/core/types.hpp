#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <variant>
#include <expected>
#include <chrono>
#include <span>

namespace atom::core {

// Type aliases
using byte_t = std::uint8_t;
using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using f32 = float;
using f64 = double;

// Time types
using TimePoint = std::chrono::high_resolution_clock::time_point;
using Duration = std::chrono::nanoseconds;

// Error handling
enum class ErrorCode {
    Success = 0,
    InvalidArgument,
    OutOfMemory,
    CudaError,
    TensorRTError,
    ModelNotFound,
    BackendNotAvailable,
    SchedulerError,
    QueueFull,
    Timeout,
    NotImplemented,
    Unknown
};

struct Error {
    ErrorCode code;
    std::string message;
    std::string file;
    int line;
    
    Error(ErrorCode c, std::string msg, std::string f = "", int l = 0)
        : code(c), message(std::move(msg)), file(std::move(f)), line(l) {}
};

template<typename T>
using Result = std::expected<T, Error>;

// Helper macro for error creation
#define ATOM_ERROR(code, msg) \
    atom::core::Error(code, msg, __FILE__, __LINE__)

// Data types
enum class DataType {
    Float32,
    Float16,
    Int32,
    Int8,
    UInt8,
    Bool
};

inline constexpr size_t DataTypeSize(DataType type) {
    switch (type) {
        case DataType::Float32: return 4;
        case DataType::Float16: return 2;
        case DataType::Int32: return 4;
        case DataType::Int8: return 1;
        case DataType::UInt8: return 1;
        case DataType::Bool: return 1;
        default: return 0;
    }
}

// Device types
enum class DeviceType {
    CPU,
    CUDA,
    AUTO
};

struct DeviceInfo {
    DeviceType type;
    int device_id;
    
    DeviceInfo(DeviceType t = DeviceType::CPU, int id = 0) 
        : type(t), device_id(id) {}
    
    bool operator==(const DeviceInfo&) const = default;
};

// Shape representation
using Shape = std::vector<i64>;

inline i64 ComputeSize(const Shape& shape) {
    i64 size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

// Priority levels for scheduling
enum class Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3
};

// Model metadata
struct ModelMetadata {
    std::string name;
    std::string version;
    std::string description;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<Shape> input_shapes;
    std::vector<Shape> output_shapes;
    std::vector<DataType> input_types;
    std::vector<DataType> output_types;
};

// Backend types
enum class BackendType {
    TensorRT,
    ONNX,
    CPU,
    Custom
};

// Inference options
struct InferenceOptions {
    DeviceInfo device{DeviceType::CUDA, 0};
    Priority priority{Priority::Normal};
    std::optional<Duration> timeout;
    bool enable_profiling{false};
    std::optional<size_t> batch_size;
};

} // namespace atom::core
