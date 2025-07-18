# CMakeLists.txt
# This file is the main CMake configuration for the Triangle Detector project.

# ---- CMake Global Configuration ----
cmake_minimum_required(VERSION 3.19)
project(triangle_detector_project VERSION 1.0.0 LANGUAGES CXX CUDA)
#set(CMAKE_BUILD_TYPE #)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_CUDA_ARCHITECTURES 89) ## Set inside CMakePresets.json
#set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc) # CUDA Path ## Set inside CMakePresets.json
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3 -DNDEBUG")
set(CMAKE_POSITION_INDEPENDENT_CODE ON) ## Library build
set(COMPILER_WARNINGS -Wall -Wextra -Wno-unused-parameter)
add_compile_options(${COMPILER_WARNINGS})   ## Show errors on production # For each target target_compile_options(<target_name> PRIVATE -Wall -Wextra)
set(DEBUG_IMAGE_DEFAULT_DIR "${CMAKE_SOURCE_DIR}/debug")
add_compile_definitions(DEBUG_IMAGE_DEFAULT_DIR="${DEBUG_IMAGE_DEFAULT_DIR}") ## Project Debug Default Configuration  #file(MAKE_DIRECTORY "${DEBUG_IMAGE_DEFAULT_DIR}") # create debug directory if it doesn't exist

## Dependencies
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/googletest)
enable_testing()
include(GoogleTest)


# ---- Projects Logs Settings ----
set(LOG_FILE_PATH "${CMAKE_SOURCE_DIR}/logs/cmake${CMAKE_SOURCE_DIR}/${CMAKE_SOURCE_DIR}/_logs.log" CACHE INTERNAL "")
set(ERROR_FILE_PATH "${CMAKE_SOURCE_DIR}/logs/cmake_errors.log" CACHE INTERNAL "")

## Log the start of the configuration
set(LOGS_DIR ${CMAKE_SOURCE_DIR}/logs)
#include(${CMAKE_SOURCE_DIR}/cmake/AddLogMessage.cmake)
#reset_log_file()
#reset_error_file()


# Check CMakePresets.json for options

# ---- Include Directories ----
include_directories(${CMAKE_SOURCE_DIR}/include) ## Project Include Directories  # For each target  target_include_directories(<target_name> PUBLIC ${CMAKE_SOURCE_DIR}/include)


# ---- Include CMake Modules ----
# Include custom CMake modules for project configuration, logging, and target management


add_library(OpenCV_Interface INTERFACE)
target_include_directories(OpenCV_Interface INTERFACE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(OpenCV_Interface INTERFACE ${OpenCV_LIBS})

add_library(OpenCV_CUDA_Interface INTERFACE)
target_include_directories(OpenCV_CUDA_Interface INTERFACE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(OpenCV_CUDA_Interface INTERFACE ${OpenCV_LIBS} opencv_cudaimgproc cuda)

# ---- Add Debug Interface ----
add_library(DebugInterface INTERFACE)


# ---- Add Shared Debug Libraries ----
add_library(debug_lib STATIC src/debug.cpp)
target_link_libraries(debug_lib PRIVATE OpenCV_Interface)
add_library(debug_cuda_lib STATIC src/debug.cpp src/debug.cu)
target_link_libraries(debug_cuda_lib PRIVATE OpenCV_CUDA_Interface)


# ---- Add Shared Libraries ----
add_library(shape_detector_logic_lib STATIC src/shared/shape_detector_logic.cpp)
target_link_libraries(shape_detector_logic_lib PRIVATE OpenCV_Interface)
add_library(shared_utils_lib STATIC src/shared/shared_utils.cpp)
target_link_libraries(shared_utils_lib PRIVATE debug_lib OpenCV_Interface DebugInterface)
add_library(shared_utils_cuda_lib STATIC src/shared/shared_utils.cpp src/shared/shared_utils.cu)
target_link_libraries(shared_utils_cuda_lib PRIVATE debug_cuda_lib OpenCV_CUDA_Interface DebugInterface)

#add_subdirectory(src)

# ---- Add Shared Libraries ----
add_subdirectory(src/cpp)
# cpp_shape_detector_lib
# cpp_utils_lib
add_subdirectory(src/cuda_opencv)
# cuda_opencv_shape_detector_lib
# cuda_opencv_utils_lib
add_subdirectory(src/pure_cuda)
# pure_cuda_shape_detector_lib
# pure_cuda_utils_lib


function(add_postbuild_clear_output TARGET_NAME)
    if(NOT ARGN MATCHES "out" OR NOT ARGN MATCHES "${CMAKE_SOURCE_DIR}")
            message(FATAL_ERROR "You request to remove a directory from path that not contains 'out' or 'output' or '${CMAKE_SOURCE_DIR}', the set path is[${ARGN}]")
    endif()
    foreach(OUTPUT_DIR IN LISTS ARGN)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E remove_directory ${OUTPUT_DIR} COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR})
    endforeach()
endfunction()



# POC Benchmark Code Files
if(ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()



if(ENABLE_EXECUTABLES)
    add_subdirectory(demo)
endif()



# ---- Tests ----


add_subdirectory(tests)






# also set the ALL_TESTS_DEPENDENCIES inside the tests/run_all_tests.cpp


# ---- Profiling ----


# ---- Banchmarks ----



