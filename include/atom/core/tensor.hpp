#pragma once

#include "types.hpp"
#include <memory>
#include <cuda_runtime.h>

namespace atom::core {

class Tensor {
public:
    // Constructors
    Tensor() = default;
    Tensor(Shape shape, DataType dtype, DeviceInfo device = DeviceInfo{DeviceType::CPU, 0});
    
    // Factory methods
    static Result<Tensor> Create(Shape shape, DataType dtype, DeviceInfo device = DeviceInfo{DeviceType::CPU, 0});
    static Result<Tensor> FromData(void* data, Shape shape, DataType dtype, DeviceInfo device, bool copy = true);
    
    // Copy and move
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    ~Tensor();
    
    // Accessors
    [[nodiscard]] const Shape& GetShape() const noexcept { return shape_; }
    [[nodiscard]] DataType GetDataType() const noexcept { return dtype_; }
    [[nodiscard]] DeviceInfo GetDevice() const noexcept { return device_; }
    [[nodiscard]] size_t GetSize() const noexcept { return size_; }
    [[nodiscard]] size_t GetByteSize() const noexcept { return size_ * DataTypeSize(dtype_); }
    [[nodiscard]] void* GetData() noexcept { return data_; }
    [[nodiscard]] const void* GetData() const noexcept { return data_; }
    [[nodiscard]] bool IsEmpty() const noexcept { return data_ == nullptr; }
    
    // Data operations
    Result<void> CopyFrom(const Tensor& src);
    Result<void> CopyTo(Tensor& dst) const;
    Result<Tensor> Clone() const;
    Result<Tensor> ToDevice(DeviceInfo device) const;
    Result<void> Fill(f32 value);
    Result<void> Zero();
    
    // Reshape (no data copy)
    Result<void> Reshape(Shape new_shape);
    
    // Data access with type checking
    template<typename T>
    Result<T*> GetDataAs() {
        if (!ValidateDataType<T>()) {
            return std::unexpected(ATOM_ERROR(ErrorCode::InvalidArgument, 
                "Type mismatch in GetDataAs"));
        }
        return static_cast<T*>(data_);
    }
    
    template<typename T>
    Result<const T*> GetDataAs() const {
        if (!ValidateDataType<T>()) {
            return std::unexpected(ATOM_ERROR(ErrorCode::InvalidArgument, 
                "Type mismatch in GetDataAs"));
        }
        return static_cast<const T*>(data_);
    }
    
private:
    Shape shape_;
    DataType dtype_{DataType::Float32};
    DeviceInfo device_{DeviceType::CPU, 0};
    void* data_{nullptr};
    size_t size_{0};
    bool owns_data_{true};
    
    Result<void> Allocate();
    void Deallocate();
    
    template<typename T>
    bool ValidateDataType() const {
        if constexpr (std::is_same_v<T, float>) {
            return dtype_ == DataType::Float32;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return dtype_ == DataType::Int32;
        } else if constexpr (std::is_same_v<T, int8_t>) {
            return dtype_ == DataType::Int8;
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            return dtype_ == DataType::UInt8;
        }
        return false;
    }
};

} // namespace atom::core
