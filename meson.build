project('atom', 'cpp', 'cuda',
  version: '1.0.0',
  default_options: [
    'cpp_std=c++20',
    'warning_level=3',
    'optimization=3',
    'b_ndebug=if-release'
  ]
)

# Compiler setup
cpp = meson.get_compiler('cpp')
cuda = meson.get_compiler('cuda')

# Find dependencies
thread_dep = dependency('threads')
opencv_dep = dependency('opencv4', required: true)
cuda_dep = dependency('cuda', version: '>=11.0', required: true)

# TensorRT dependencies
tensorrt_inc = include_directories('/usr/include/x86_64-linux-gnu')
tensorrt_lib_dir = '/usr/lib/x86_64-linux-gnu'

nvinfer_dep = cpp.find_library('nvinfer', dirs: [tensorrt_lib_dir], required: true)
nvonnxparser_dep = cpp.find_library('nvonnxparser', dirs: [tensorrt_lib_dir], required: true)
nvinfer_plugin_dep = cpp.find_library('nvinfer_plugin', dirs: [tensorrt_lib_dir], required: true)
nvparsers_dep = cpp.find_library('nvparsers', dirs: [tensorrt_lib_dir], required: true)

cudart_dep = cpp.find_library('cudart', dirs: ['/usr/local/cuda/lib64'], required: true)
cublas_dep = cpp.find_library('cublas', dirs: ['/usr/local/cuda/lib64'], required: true)
cudnn_dep = cpp.find_library('cudnn', dirs: ['/usr/lib/x86_64-linux-gnu'], required: true)

# Include directories
inc_dirs = include_directories('include')
cuda_inc = include_directories('/usr/local/cuda/include')

# Collect all dependencies
atom_deps = [
  thread_dep,
  opencv_dep,
  cuda_dep,
  nvinfer_dep,
  nvonnxparser_dep,
  nvinfer_plugin_dep,
  nvparsers_dep,
  cudart_dep,
  cublas_dep,
  cudnn_dep
]

# Core library sources
core_sources = [
  'src/core/config.cpp',
  'src/core/tensor.cpp',
  'src/core/model_factory.cpp',
  'src/core/model_manager.cpp',
  'src/core/model_wrapper.cpp'
]

# Inference backend sources
inference_sources = [
  'src/inference/backend.cpp',
  'src/inference/backend_factory.cpp',
  'src/inference/tensorrt_backend.cpp',
  'src/inference/onnx_backend.cpp',
  'src/inference/cpu_backend.cpp'
]

# Scheduler sources
scheduler_sources = [
  'src/scheduler/scheduler.cpp',
  'src/scheduler/task.cpp',
  'src/scheduler/thread_pool.cpp',
  'src/scheduler/dependency_graph.cpp'
]

# Logging sources
logging_sources = [
  'src/logging/logger.cpp',
  'src/logging/metrics.cpp'
]

# Data sources
data_sources = [
  'src/data/pipeline.cpp',
  'src/data/queue.cpp',
  'src/data/preprocessor.cpp'
]

# Visualization sources
viz_sources = [
  'src/viz/dashboard.cpp',
  'src/viz/profiler.cpp'
]

# Build main atom library
atom_lib = library('atom',
  core_sources + inference_sources + scheduler_sources + 
  logging_sources + data_sources + viz_sources,
  include_directories: [inc_dirs, cuda_inc, tensorrt_inc],
  dependencies: atom_deps,
  install: true
)

atom_dep = declare_dependency(
  link_with: atom_lib,
  include_directories: [inc_dirs, cuda_inc, tensorrt_inc],
  dependencies: atom_deps
)

# Build models
subdir('models')

# Build examples
subdir('examples')

# Install headers
install_subdir('include/atom', install_dir: 'include')
