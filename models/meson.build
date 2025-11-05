# Build detection models
yolo_lib = library('yolo',
  'detection/yolo/yolov8.cpp',
  include_directories: [inc_dirs, cuda_inc, tensorrt_inc],
  dependencies: [atom_dep],
  install: true
)

# Build classification models
resnet_lib = library('resnet',
  'classification/resnet/resnet50.cpp',
  include_directories: [inc_dirs, cuda_inc, tensorrt_inc],
  dependencies: [atom_dep],
  install: true
)

# You can add more models here following the same pattern
