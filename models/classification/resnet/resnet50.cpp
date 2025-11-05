#include "resnet50.hpp"
#include <atom/core/model_factory.hpp>

// Register ResNet50 model with the factory
REGISTER_MODEL(atom::models::ResNet50, "resnet50");
