#include <atom/core/model_manager.hpp>
#include <atom/core/model_factory.hpp>
#include <atom/scheduler/scheduler.hpp>
#include <atom/data/preprocessor.hpp>
#include <atom/logging/logger.hpp>
#include <iostream>

int main() {
    using namespace atom;
    
    // Initialize logger
    logging::Logger::Instance().SetLevel(logging::LogLevel::Info);
    logging::Logger::Instance().EnableConsoleOutput(true);
    
    LOG_INFO("Starting Atom Multi-Model Inference Example");
    
    try {
        // Create scheduler
        scheduler::SchedulerConfig config;
        config.num_threads = 4;
        scheduler::Scheduler sched(config);
        
        auto start_result = sched.Start();
        if (!start_result) {
            LOG_ERROR("Failed to start scheduler: " + start_result.error().message);
            return 1;
        }
        
        LOG_INFO("Scheduler started with " + std::to_string(config.num_threads) + " threads");
        
        // Load models
        auto& model_mgr = core::ModelManager::Instance();
        
        core::InferenceOptions yolo_options;
        yolo_options.device = core::DeviceInfo{core::DeviceType::CUDA, 0};
        yolo_options.priority = core::Priority::High;
        
        auto yolo_result = model_mgr.LoadModel(
            "yolo_detector",
            "yolov8",
            "/path/to/yolov8.engine",
            yolo_options
        );
        
        if (yolo_result) {
            LOG_INFO("YOLOv8 model loaded successfully");
        } else {
            LOG_WARNING("Failed to load YOLOv8: " + yolo_result.error().message);
        }
        
        core::InferenceOptions resnet_options;
        resnet_options.device = core::DeviceInfo{core::DeviceType::CUDA, 0};
        resnet_options.priority = core::Priority::Normal;
        
        auto resnet_result = model_mgr.LoadModel(
            "resnet_classifier",
            "resnet50",
            "/path/to/resnet50.engine",
            resnet_options
        );
        
        if (resnet_result) {
            LOG_INFO("ResNet50 model loaded successfully");
        } else {
            LOG_WARNING("Failed to load ResNet50: " + resnet_result.error().message);
        }
        
        // Create dummy input tensors
        auto input_tensor = core::Tensor::Create(
            {1, 3, 640, 640},
            core::DataType::Float32,
            core::DeviceInfo{core::DeviceType::CUDA, 0}
        );
        
        if (!input_tensor) {
            LOG_ERROR("Failed to create input tensor");
            return 1;
        }
        
        // Submit inference tasks
        std::vector<scheduler::TaskId> task_ids;
        
        for (int i = 0; i < 10; ++i) {
            auto model = model_mgr.GetModel("yolo_detector");
            if (model) {
                auto task_id = sched.SubmitTask(
                    *model,
                    {*input_tensor},
                    core::Priority::High,
                    [i](const scheduler::TaskResult& result) {
                        if (result.status == scheduler::TaskStatus::Completed) {
                            LOG_INFO("Task " + std::to_string(i) + " completed successfully");
                        } else {
                            LOG_ERROR("Task " + std::to_string(i) + " failed");
                        }
                    }
                );
                
                if (task_id) {
                    task_ids.push_back(*task_id);
                    LOG_DEBUG("Submitted task " + std::to_string(*task_id));
                }
            }
        }
        
        LOG_INFO("Submitted " + std::to_string(task_ids.size()) + " tasks");
        
        // Wait for all tasks to complete
        auto results = sched.WaitForAll(task_ids);
        
        if (results) {
            LOG_INFO("All tasks completed");
            
            size_t successful = 0;
            for (const auto& result : *results) {
                if (result.status == scheduler::TaskStatus::Completed) {
                    successful++;
                }
            }
            
            LOG_INFO("Successful tasks: " + std::to_string(successful) + "/" + 
                    std::to_string(results->size()));
        }
        
        // Print statistics
        const auto& stats = sched.GetStatistics();
        LOG_INFO("Total tasks: " + std::to_string(stats.total_tasks.load()));
        LOG_INFO("Completed tasks: " + std::to_string(stats.completed_tasks.load()));
        LOG_INFO("Average execution time: " + 
                std::to_string(stats.GetAverageExecutionTimeMs()) + " ms");
        
        // Cleanup
        sched.Stop();
        model_mgr.UnloadAll();
        
        LOG_INFO("Atom Multi-Model Inference Example completed successfully");
        
    } catch (const std::exception& e) {
        LOG_CRITICAL("Exception: " + std::string(e.what()));
        return 1;
    }
    
    return 0;
}
