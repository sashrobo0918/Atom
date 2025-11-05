#pragma once

#include "../core/types.hpp"
#include "../core/tensor.hpp"
#include <opencv2/opencv.hpp>
#include <functional>

namespace atom::data {

// Preprocessing operations
struct PreprocessConfig {
    cv::Size target_size{640, 640};
    cv::Scalar mean{0.485, 0.456, 0.406};
    cv::Scalar std{0.229, 0.224, 0.225};
    bool normalize{true};
    bool to_rgb{true};
    cv::InterpolationFlags interpolation{cv::INTER_LINEAR};
};

class Preprocessor {
public:
    explicit Preprocessor(PreprocessConfig config = PreprocessConfig{});
    
    // Image preprocessing
    atom::core::Result<atom::core::Tensor> PreprocessImage(const cv::Mat& image) const;
    atom::core::Result<std::vector<atom::core::Tensor>> PreprocessBatch(
        const std::vector<cv::Mat>& images) const;
    
    // Custom preprocessing pipeline
    using PreprocessFunc = std::function<atom::core::Result<cv::Mat>(const cv::Mat&)>;
    void AddCustomStep(PreprocessFunc func);
    void ClearCustomSteps();
    
    // Utilities
    static atom::core::Result<cv::Mat> LoadImage(const std::string& path);
    static atom::core::Result<void> SaveImage(const std::string& path, const cv::Mat& image);
    static atom::core::Result<cv::Mat> TensorToMat(const atom::core::Tensor& tensor);
    static atom::core::Result<atom::core::Tensor> MatToTensor(const cv::Mat& mat, 
        atom::core::DeviceInfo device = atom::core::DeviceInfo{atom::core::DeviceType::CPU, 0});
    
private:
    PreprocessConfig config_;
    std::vector<PreprocessFunc> custom_steps_;
    
    atom::core::Result<cv::Mat> ApplyPreprocessing(const cv::Mat& image) const;
};

} // namespace atom::data
