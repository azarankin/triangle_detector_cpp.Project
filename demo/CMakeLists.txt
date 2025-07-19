# /demo/CMakeLists.txt

# This file is part of the Triangle Detector Executable.
add_library(shape_detector_logic_lib STATIC src/shared/shape_detector_logic.cpp)
target_link_libraries(shape_detector_logic_lib PRIVATE OpenCV_Interface)

# CPP demo executable
add_executable(triangle_detector_cpu_logic STATIC demo_cpp.cpp)
target_link_libraries(triangle_detector_cpu_logic PRIVATE shape_detector_logic_lib cpp_shape_detector_lib)
target_include_directories(triangle_detector_cpu_logic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_postbuild_clear_output(triangle_detector_cpu_logic ${CMAKE_CURRENT_SOURCE_DIR}/out)