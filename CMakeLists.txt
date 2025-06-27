cmake_minimum_required(VERSION 3.19) 


project(triangle_detector LANGUAGES CXX CUDA)


# ---- Global Configuration ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
### Set in CMakePresets.json
#set(CMAKE_CUDA_ARCHITECTURES 89)
#set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc) # CUDA Path

# ---- Dependencies ----
find_package(OpenCV REQUIRED)
add_subdirectory(third_party/googletest)
enable_testing()
include(GoogleTest)

# ---- Utility: Post-build Cleanup ----
function(add_postbuild_clear_output TARGET_NAME OUTPUT_DIR)
  add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
  )
endfunction()


# ---- Libraries ----

# ---- C++ Code Files----
add_library(triangle_detector_cpu shape_detector_cpp.cpp)
target_include_directories(triangle_detector_cpu PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(triangle_detector_cpu PRIVATE ${OpenCV_LIBS})
# ---- CUDA OpenCV Code Files ----
add_library(triangle_detector_cuda shape_detector_cuda_opencv.cu)
target_include_directories(triangle_detector_cuda PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(triangle_detector_cuda PRIVATE ${OpenCV_LIBS} cuda)


# ---- Executables ----

## CPU Demo
add_executable(triangle_detector_cpp_demo demo/demo_cpp.cpp)
target_link_libraries(triangle_detector_cpp_demo PRIVATE triangle_detector_cpu)
add_postbuild_clear_output(triangle_detector_cpp_demo ${CMAKE_SOURCE_DIR}/demo/out)


## CPU Tests
add_executable(triangle_detector_cpp_tests test/triangle_cpp_test.cpp)
target_link_libraries(triangle_detector_cpp_tests
  PRIVATE gtest gtest_main triangle_detector_cpu ${OpenCV_LIBS}
)
gtest_discover_tests(triangle_detector_cpp_tests)
add_postbuild_clear_output(triangle_detector_cpp_tests ${CMAKE_SOURCE_DIR}/test/current_output)


## CUDA Tests
add_executable(triangle_detector_cuda_tests test/triangle_cuda_opencv_test.cpp)
target_link_libraries(triangle_detector_cuda_tests
  PRIVATE gtest gtest_main triangle_detector_cuda ${OpenCV_LIBS} cuda
)
gtest_discover_tests(triangle_detector_cuda_tests)
add_postbuild_clear_output(triangle_detector_cuda_tests ${CMAKE_SOURCE_DIR}/test/current_output)


## All Tests Runner
add_executable(run_all_tests test/run_all_tests.cpp)
add_dependencies(run_all_tests triangle_detector_cpp_tests triangle_detector_cuda_tests)
set_target_properties(run_all_tests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)