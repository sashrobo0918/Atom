#include <atom/core/model_manager.hpp>
#include <atom/scheduler/scheduler.hpp>
#include <atom/data/preprocessor.hpp>
#include <atom/logging/logger.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    using namespace atom;
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path>" << std::endl;
        return 1;
    }
    
    logging::Logger::Instance().SetLevel(logging::LogLevel::Info);
    LOG_INFO("Starting Real-time Video Processing Example");
    
    try {
        // Open video
        cv::VideoCapture cap(argv[1]);
        if (!cap.isOpened()) {
            LOG_ERROR("Failed to open video: " + std::string(argv[1]));
            return 1;
        }
        
        LOG_INFO("Video opened: " + std::to_string(cap.get(cv::CAP_PROP_FRAME_WIDTH)) + 
                "x" + std::to_string(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
        
        // Initialize scheduler
        scheduler::SchedulerConfig config;
        config.num_threads = 2;
        scheduler::Scheduler sched(config);
        sched.Start();
        
        // Load detection model
        auto& model_mgr = core::ModelManager::Instance();
        core::InferenceOptions options;
        options.device = core::DeviceInfo{core::DeviceType::CUDA, 0};
        
        auto load_result = model_mgr.LoadModel(
            "detector",
            "yolov8",
            "/path/to/yolov8.engine",
            options
        );
        
        if (!load_result) {
            LOG_ERROR("Failed to load model");
            return 1;
        }
        
        auto model = model_mgr.GetModel("detector");
        if (!model) {
            LOG_ERROR("Failed to get model");
            return 1;
        }
        
        // Create preprocessor
        data::PreprocessConfig preprocess_config;
        preprocess_config.target_size = cv::Size(640, 640);
        data::Preprocessor preprocessor(preprocess_config);
        
        // Process video frames
        cv::Mat frame;
        int frame_count = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        while (cap.read(frame)) {
            frame_count++;
            
            // Preprocess frame
            auto tensor = preprocessor.PreprocessImage(frame);
            if (!tensor) {
                LOG_WARNING("Failed to preprocess frame " + std::to_string(frame_count));
                continue;
            }
            
            // Submit inference task
            auto task_id = sched.SubmitTask(
                *model,
                {*tensor},
                core::Priority::Normal,
                [frame_count](const scheduler::TaskResult& result) {
                    if (result.status == scheduler::TaskStatus::Completed) {
                        LOG_DEBUG("Frame " + std::to_string(frame_count) + " processed");
                    }
                }
            );
            
            if (!task_id) {
                LOG_WARNING("Failed to submit task for frame " + std::to_string(frame_count));
            }
            
            // Limit to 30 FPS
            if (frame_count % 30 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration<double>(now - start_time).count();
                double fps = frame_count / elapsed;
                LOG_INFO("Processed " + std::to_string(frame_count) + 
                        " frames, FPS: " + std::to_string(fps));
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double>(end_time - start_time).count();
        double fps = frame_count / elapsed;
        
        LOG_INFO("Total frames processed: " + std::to_string(frame_count));
        LOG_INFO("Average FPS: " + std::to_string(fps));
        
        // Cleanup
        sched.Stop();
        model_mgr.UnloadAll();
        
        LOG_INFO("Real-time Video Processing Example completed");
        
    } catch (const std::exception& e) {
        LOG_CRITICAL("Exception: " + std::string(e.what()));
        return 1;
    }
    
    return 0;
}
