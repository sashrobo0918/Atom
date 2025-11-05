if get_option('enable_examples')
  # Multi-model inference example
  executable('multi_model_inference',
    'multi_model_inference.cpp',
    include_directories: [inc_dirs, cuda_inc, tensorrt_inc],
    dependencies: [atom_dep],
    link_with: [yolo_lib, resnet_lib],
    install: false
  )
  
  # Real-time video processing example
  executable('realtime_video_processing',
    'realtime_video_processing.cpp',
    include_directories: [inc_dirs, cuda_inc, tensorrt_inc],
    dependencies: [atom_dep, opencv_dep],
    link_with: [yolo_lib],
    install: false
  )
endif
