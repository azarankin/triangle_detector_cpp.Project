cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(triangle_detector LANGUAGES CXX CUDA)

# ---- OpenCV ----
find_package(OpenCV REQUIRED)

# ---- Code Files----
add_library(triangle_detector shape_detector_cpp.cpp)

set_property(TARGET triangle_detector PROPERTY CXX_STANDARD 20)

# ---- include header files ----
target_include_directories(triangle_detector PUBLIC ${OpenCV_INCLUDE_DIRS})

# ---- linking libraries to my project ----
target_link_libraries(triangle_detector PRIVATE ${OpenCV_LIBS} cuda)




# ---- Main Executable ----
add_executable(triangle_detector_executable demo_cpp.cpp)
target_link_libraries(triangle_detector_executable PRIVATE triangle_detector)


# --- GoogleTest integration ---
add_subdirectory(third_party/googletest)
enable_testing()

add_executable(triangle_detector_tests test/triangle_cpp_test.cpp)
# ---- linking libraries to my project ----
target_link_libraries(triangle_detector_tests PRIVATE gtest gtest_main ${OpenCV_LIBS} cuda triangle_detector)
#Each test file sohuld #include <gtest/gtest.h>

include(GoogleTest)
gtest_discover_tests(triangle_detector_tests)

# Run the test on CMake build
add_custom_command(TARGET triangle_detector_tests
  POST_BUILD
  COMMAND $<TARGET_FILE:triangle_detector_tests>
  #delete test directory /test/current_output
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/test/current_output
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/test/current_output
)





# ---- headers for my project ----
#target_include_directories(triangle_detector PRIVATE ${OpenCV_INCLUDE_DIRS})
