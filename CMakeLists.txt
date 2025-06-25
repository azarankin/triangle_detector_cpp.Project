cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(triangle_detector LANGUAGES CXX CUDA)


# ---- OpenCV ----
find_package(OpenCV REQUIRED)

# ---- Code Files----
add_executable(triangle_detector main.cpp shape_detector.cpp)

set_property(TARGET triangle_detector PROPERTY CXX_STANDARD 20)

# ---- linking libraries to my project ----
target_link_libraries(triangle_detector PRIVATE ${OpenCV_LIBS} cuda)

# ---- headers for my project ----
#target_include_directories(triangle_detector PRIVATE ${OpenCV_INCLUDE_DIRS})
