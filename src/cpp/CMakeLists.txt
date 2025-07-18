# /src/cpp/CMakeLists.txt

add_library(cpp_utils_lib STATIC utils_cpp.cpp)
target_link_libraries(cpp_utils_lib PUBLIC shared_utils_lib OpenCV_Interface DebugInterface)

add_library(cpp_shape_detector_lib STATIC shape_detector_cpp.cpp)
target_link_libraries(cpp_shape_detector_lib PRIVATE cpp_utils_lib)
