
# /src/CMakeLists.txt
# ---- Add Shared Libraries for C++ and CUDA ----


# This file is part of the Shape Detector Debug Utilities.
add_library(debug_lib STATIC debug.cpp)
target_link_libraries(debug_lib PRIVATE OpenCV_Interface)
add_library(debug_cuda_lib STATIC debug.cpp debug.cu)
target_link_libraries(debug_cuda_lib PRIVATE OpenCV_CUDA_Interface)



# This file is part of the Shape Detector Shared C++ Utilities.
add_library(shared_utils_lib STATIC shared/shared_utils.cpp)
target_link_libraries(shared_utils_lib PRIVATE debug_lib OpenCV_Interface DebugInterface)
# This file is part of the Shape Detector Shared CUDF Utilities.
add_library(shared_utils_cuda_lib STATIC shared/shared_utils.cpp shared/shared_utils.cu)
target_link_libraries(shared_utils_cuda_lib PRIVATE debug_cuda_lib OpenCV_CUDA_Interface DebugInterface)


# This file is part of the Shape Detector C++ project.
add_library(cpp_utils_lib STATIC cpp/utils_cpp.cpp)
target_link_libraries(cpp_utils_lib PUBLIC shared_utils_lib OpenCV_Interface DebugInterface ProfilingInterface)

add_library(cpp_shape_detector_lib STATIC cpp/shape_detector_cpp.cpp)
target_link_libraries(cpp_shape_detector_lib PRIVATE cpp_utils_lib ProfilingInterface)



# This file is part of the Shape Detector OpenCV CUDA project.
add_library(cuda_opencv_utils_lib STATIC cuda_opencv/utils_cuda_opencv.cu)
target_link_libraries(cuda_opencv_utils_lib PUBLIC shared_utils_cuda_lib OpenCV_CUDA_Interface DebugInterface ProfilingInterface)

add_library(cuda_opencv_shape_detector_lib STATIC cuda_opencv/shape_detector_cuda_opencv.cu)
target_link_libraries(cuda_opencv_shape_detector_lib PRIVATE cuda_opencv_utils_lib ProfilingInterface)



# This file is part of the Shape Detector Pure CUDA project.
add_library(pure_cuda_utils_lib STATIC pure_cuda/utils_pure_cuda.cu)
target_link_libraries(pure_cuda_utils_lib PUBLIC shared_utils_cuda_lib OpenCV_CUDA_Interface DebugInterface ProfilingInterface)

add_library(pure_cuda_shape_detector_lib STATIC pure_cuda/shape_detector_pure_cuda.cu)
target_link_libraries(pure_cuda_shape_detector_lib PRIVATE pure_cuda_utils_lib ProfilingInterface)