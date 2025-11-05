#include "yolov8.hpp"
#include <atom/core/model_factory.hpp>

// Register YOLOv8 model with the factory
REGISTER_MODEL(atom::models::YOLOv8, "yolov8");
