cmake_minimum_required(VERSION 3.19) 

# --- CUDA Global Architecture ---
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc) # CUDA Path
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)


project(triangle_detector LANGUAGES CXX CUDA)


# ---- OpenCV ----
find_package(OpenCV REQUIRED)


# --- GoogleTest ---
add_subdirectory(third_party/googletest)
enable_testing()

# ---- C++ Code Files----
add_library(triangle_detector_cpu
    shape_detector_cpp.cpp
#    shape_detector_cpp.h
)
target_include_directories(triangle_detector_cpu PUBLIC ${OpenCV_INCLUDE_DIRS}) # include header files
target_link_libraries(triangle_detector_cpu PRIVATE ${OpenCV_LIBS}) # linking libraries to project


# ---- CUDA OpenCV Code Files ----
add_library(triangle_detector_cuda
    shape_detector_cuda_opencv.cu
#    shape_detector_cuda_opencv.cuh
)
target_include_directories(triangle_detector_cuda PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(triangle_detector_cuda PRIVATE ${OpenCV_LIBS} cuda)


# ---- Executables ----
# C++ Demo Executable
add_executable(triangle_detector_cpp_demo demo/demo_cpp.cpp)
target_link_libraries(triangle_detector_cpp_demo PRIVATE triangle_detector_cpu)


# ---- Custom Executable Commadns ----
# Reset the "current_output" directory
add_custom_command(TARGET triangle_detector_cpp_demo POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/demo/out
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/demo/out
)


# ---- CPU Tests ----
add_executable(triangle_detector_cpp_tests test/triangle_cpp_test.cpp)
target_link_libraries(triangle_detector_cpp_tests
    PRIVATE gtest gtest_main triangle_detector_cpu ${OpenCV_LIBS}
) # linking libraries to test project
# Each test file sohuld #include <gtest/gtest.h>
include(GoogleTest)
gtest_discover_tests(triangle_detector_cpp_tests)


# ---- Tests Custom Commadns ----
# Reset the "current_output" directory
add_custom_command(TARGET triangle_detector_cpp_tests POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/test/current_output
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/test/current_output
)


# ---- CUDA OpenCV Tests ----
add_executable(triangle_detector_cuda_tests test/triangle_cuda_opencv_test.cpp)
target_link_libraries(triangle_detector_cuda_tests
    PRIVATE gtest gtest_main triangle_detector_cuda ${OpenCV_LIBS} cuda
)
include(GoogleTest)
gtest_discover_tests(triangle_detector_cuda_tests)



add_executable(run_all_tests test/run_all_tests.cpp)
add_dependencies(run_all_tests triangle_detector_cpp_tests triangle_detector_cuda_tests)
set_target_properties(run_all_tests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)