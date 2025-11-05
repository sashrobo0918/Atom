#include "atom/core/tensor.hpp"
#include <cstring>
#include <stdexcept>

namespace atom::core {

Tensor::Tensor(Shape shape, DataType dtype, DeviceInfo device)
    : shape_(std::move(shape)), dtype_(dtype), device_(device), owns_data_(true) {
    size_ = ComputeSize(shape_);
    auto result = Allocate();
    if (!result) {
        throw std::runtime_error(result.error().message);
    }
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), dtype_(other.dtype_), device_(other.device_),
      size_(other.size_), owns_data_(true) {
    auto result = Allocate();
    if (!result) {
        throw std::runtime_error(result.error().message);
    }
    CopyFrom(other);
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        Deallocate();
        shape_ = other.shape_;
        dtype_ = other.dtype_;
        device_ = other.device_;
        size_ = other.size_;
        owns_data_ = true;
        
        auto result = Allocate();
        if (!result) {
            throw std::runtime_error(result.error().message);
        }
        CopyFrom(other);
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), dtype_(other.dtype_),
      device_(other.device_), data_(other.data_), size_(other.size_),
      owns_data_(other.owns_data_) {
    other.data_ = nullptr;
    other.owns_data_ = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        Deallocate();
        
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        device_ = other.device_;
        data_ = other.data_;
        size_ = other.size_;
        owns_data_ = other.owns_data_;
        
        other.data_ = nullptr;
        other.owns_data_ = false;
    }
    return *this;
}

Tensor::~Tensor() {
    Deallocate();
}

Result<Tensor> Tensor::Create(Shape shape, DataType dtype, DeviceInfo device) {
    try {
        return Tensor(std::move(shape), dtype, device);
    } catch (const std::exception& e) {
        return std::unexpected(ATOM_ERROR(ErrorCode::OutOfMemory, e.what()));
    }
}

Result<Tensor> Tensor::FromData(void* data, Shape shape, DataType dtype, 
                                DeviceInfo device, bool copy) {
    Tensor tensor;
    tensor.shape_ = std::move(shape);
    tensor.dtype_ = dtype;
    tensor.device_ = device;
    tensor.size_ = ComputeSize(tensor.shape_);
    
    if (copy) {
        tensor.owns_data_ = true;
        auto result = tensor.Allocate();
        if (!result) return std::unexpected(result.error());
        
        const size_t byte_size = tensor.GetByteSize();
        if (device.type == DeviceType::CPU) {
            std::memcpy(tensor.data_, data, byte_size);
        } else {
            cudaError_t err = cudaMemcpy(tensor.data_, data, byte_size, cudaMemcpyDefault);
            if (err != cudaSuccess) {
                return std::unexpected(ATOM_ERROR(ErrorCode::CudaError, 
                    "CUDA memcpy failed: " + std::string(cudaGetErrorString(err))));
            }
        }
    } else {
        tensor.data_ = data;
        tensor.owns_data_ = false;
    }
    
    return tensor;
}

Result<void> Tensor::Allocate() {
    const size_t byte_size = GetByteSize();
    
    if (device_.type == DeviceType::CPU) {
        data_ = std::malloc(byte_size);
        if (!data_) {
            return std::unexpected(ATOM_ERROR(ErrorCode::OutOfMemory, 
                "Failed to allocate CPU memory"));
        }
    } else if (device_.type == DeviceType::CUDA) {
        cudaError_t err = cudaMalloc(&data_, byte_size);
        if (err != cudaSuccess) {
            return std::unexpected(ATOM_ERROR(ErrorCode::CudaError, 
                "CUDA allocation failed: " + std::string(cudaGetErrorString(err))));
        }
    }
    
    return {};
}

void Tensor::Deallocate() {
    if (data_ && owns_data_) {
        if (device_.type == DeviceType::CPU) {
            std::free(data_);
        } else if (device_.type == DeviceType::CUDA) {
            cudaFree(data_);
        }
        data_ = nullptr;
    }
}

Result<void> Tensor::CopyFrom(const Tensor& src) {
    if (size_ != src.size_ || dtype_ != src.dtype_) {
        return std::unexpected(ATOM_ERROR(ErrorCode::InvalidArgument, 
            "Tensor dimensions or types do not match"));
    }
    
    const size_t byte_size = GetByteSize();
    cudaMemcpyKind kind;
    
    if (device_.type == DeviceType::CPU && src.device_.type == DeviceType::CPU) {
        std::memcpy(data_, src.data_, byte_size);
        return {};
    } else if (device_.type == DeviceType::CUDA && src.device_.type == DeviceType::CPU) {
        kind = cudaMemcpyHostToDevice;
    } else if (device_.type == DeviceType::CPU && src.device_.type == DeviceType::CUDA) {
        kind = cudaMemcpyDeviceToHost;
    } else {
        kind = cudaMemcpyDeviceToDevice;
    }
    
    cudaError_t err = cudaMemcpy(data_, src.data_, byte_size, kind);
    if (err != cudaSuccess) {
        return std::unexpected(ATOM_ERROR(ErrorCode::CudaError, 
            "CUDA copy failed: " + std::string(cudaGetErrorString(err))));
    }
    
    return {};
}

Result<void> Tensor::CopyTo(Tensor& dst) const {
    return dst.CopyFrom(*this);
}

Result<Tensor> Tensor::Clone() const {
    auto tensor = Create(shape_, dtype_, device_);
    if (!tensor) return std::unexpected(tensor.error());
    
    auto result = tensor->CopyFrom(*this);
    if (!result) return std::unexpected(result.error());
    
    return tensor;
}

Result<Tensor> Tensor::ToDevice(DeviceInfo device) const {
    if (device == device_) {
        return Clone();
    }
    
    auto tensor = Create(shape_, dtype_, device);
    if (!tensor) return std::unexpected(tensor.error());
    
    auto result = tensor->CopyFrom(*this);
    if (!result) return std::unexpected(result.error());
    
    return tensor;
}

Result<void> Tensor::Fill(f32 value) {
    // Simplified implementation
    if (device_.type == DeviceType::CPU && dtype_ == DataType::Float32) {
        float* data = static_cast<float*>(data_);
        for (size_t i = 0; i < size_; ++i) {
            data[i] = value;
        }
        return {};
    }
    
    return std::unexpected(ATOM_ERROR(ErrorCode::NotImplemented, 
        "Fill not implemented for this device/dtype"));
}

Result<void> Tensor::Zero() {
    const size_t byte_size = GetByteSize();
    
    if (device_.type == DeviceType::CPU) {
        std::memset(data_, 0, byte_size);
    } else if (device_.type == DeviceType::CUDA) {
        cudaError_t err = cudaMemset(data_, 0, byte_size);
        if (err != cudaSuccess) {
            return std::unexpected(ATOM_ERROR(ErrorCode::CudaError, 
                "CUDA memset failed: " + std::string(cudaGetErrorString(err))));
        }
    }
    
    return {};
}

Result<void> Tensor::Reshape(Shape new_shape) {
    const i64 new_size = ComputeSize(new_shape);
    if (new_size != static_cast<i64>(size_)) {
        return std::unexpected(ATOM_ERROR(ErrorCode::InvalidArgument, 
            "New shape must have the same number of elements"));
    }
    
    shape_ = std::move(new_shape);
    return {};
}

} // namespace atom::core
