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

set(CMAKE_POSITION_INDEPENDENT_CODE ON) ## Library build
set(COMPILER_WARNINGS -Wall -Wextra -Wno-unused-parameter)
add_compile_options(${COMPILER_WARNINGS})   ## Show errors on production # For each target target_compile_options(<target_name> PRIVATE -Wall -Wextra)
set(DEBUG_IMAGE_DEFAULT_DIR "${CMAKE_SOURCE_DIR}/debug")
add_compile_definitions(DEBUG_IMAGE_DEFAULT_DIR="${DEBUG_IMAGE_DEFAULT_DIR}") ## Define default debug directory
file(MAKE_DIRECTORY "${DEBUG_IMAGE_DEFAULT_DIR}") ## Create debug directory if it doesn't exist



## Dependencies
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/googletest)
enable_testing()
include(GoogleTest)

# ---- Include Directories ----
include_directories(${CMAKE_SOURCE_DIR}/include) ## Project Include Directories  # For each target  target_include_directories(<target_name> PUBLIC ${CMAKE_SOURCE_DIR}/include)


# ---- Interfaces ----
# ---- Add OpenCV Interface ----
add_library(OpenCV_Interface INTERFACE)
target_include_directories(OpenCV_Interface INTERFACE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(OpenCV_Interface INTERFACE ${OpenCV_LIBS})

add_library(OpenCV_CUDA_Interface INTERFACE)
target_include_directories(OpenCV_CUDA_Interface INTERFACE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(OpenCV_CUDA_Interface INTERFACE ${OpenCV_LIBS} opencv_cudaimgproc cuda)


# ---- CMake Functions ----
function(add_postbuild_clear_output TARGET_NAME)
    if(NOT ARGN MATCHES "out" OR NOT ARGN MATCHES "${CMAKE_SOURCE_DIR}")
            message(FATAL_ERROR "You request to remove a directory from path that not contains 'out' or 'output' or '${CMAKE_SOURCE_DIR}', the set path is[${ARGN}]")
    endif()
    foreach(OUTPUT_DIR IN LISTS ARGN)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E remove_directory ${OUTPUT_DIR} COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR})
    endforeach()
endfunction()


# ---- Add Debug Interface ----
add_library(DebugInterface INTERFACE)
if(ENABLE_IMAGE_DEBUG AND NOT DEFINED SWITCH_TO_PROFILING)
    set(DEBUG_IMAGE_DIRECTORY "${CMAKE_SOURCE_DIR}/test/data/current_output/debug")
    file(MAKE_DIRECTORY "${DEBUG_IMAGE_DIRECTORY}")
    set(DEBUG_DEFINES
        ENABLE_DEBUG_IMAGES
        DEBUG_IMAGE_DIR="${DEBUG_IMAGE_DIRECTORY}"
    )
    target_compile_definitions(DebugInterface INTERFACE ${DEBUG_DEFINES})
    # add_postbuild_clear_output(DebugInterface ${DEBUG_IMAGE_DIRECTORY})
endif()


# ---- Add Profiling Interface ----
add_library(ProfilingInterface INTERFACE)
if(SWITCH_TO_PROFILING)
    set(PROFILING_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/profiling/output_results")
    set(PROFILING_DEFINES
        ENABLE_PROFILING
        ENABLE_NVTX
        PROFILING_OUTPUT_DIRECTORY="${PROFILING_OUTPUT_DIRECTORY}"
    )
    file(MAKE_DIRECTORY "${PROFILING_OUTPUT_DIRECTORY}")
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        list(APPEND PROFILING_DEFINES COMPILED_AS_RELEASE)
        if(CMAKE_CXX_FLAGS_RELEASE MATCHES "-O3")
            list(APPEND PROFILING_DEFINES COMPILED_WITH_O3)
        else()
            message(FATAL_ERROR "==> Warning: Release build *without* -O3!")
        endif()
    else()
        message(FATAL_ERROR "Profiling must be built in Release or RelWithDebInfo mode! Current: ${CMAKE_BUILD_TYPE}")
    endif()
    target_compile_definitions(ProfilingInterface INTERFACE ${PROFILING_DEFINES})
endif()




# ---- Add Shared Libraries ----
add_subdirectory(src)
# debug_lib
# debug_cuda_lib
# shared_utils_lib
# shared_utils_cuda_lib
# cpp_shape_detector_lib
# cpp_utils_lib
# cuda_opencv_shape_detector_lib
# cuda_opencv_utils_lib
# pure_cuda_shape_detector_lib
# pure_cuda_utils_lib


if(SWITCH_TO_PROFILING)
    # ---- Profiling ----
    add_subdirectory(profiling)
else()
    
    if(ENABLE_BENCHMARKS)
        # ---- Benchmarks ----
        add_subdirectory(benchmarks)
    endif()

    if(ENABLE_EXECUTABLES)
        # ---- Executables ----
        add_subdirectory(demo)
    endif()

    # ---- Tests ----
    add_subdirectory(tests)

endif()





# ---- Projects Logs Settings ----
#set(LOG_FILE_PATH "${CMAKE_SOURCE_DIR}/logs/cmake${CMAKE_SOURCE_DIR}/${CMAKE_SOURCE_DIR}/_logs.log" CACHE INTERNAL "")
#set(ERROR_FILE_PATH "${CMAKE_SOURCE_DIR}/logs/cmake_errors.log" CACHE INTERNAL "")

## Log the start of the configuration
#set(LOGS_DIR ${CMAKE_SOURCE_DIR}/logs)
#include(${CMAKE_SOURCE_DIR}/cmake/AddLogMessage.cmake)
#reset_log_file()
#reset_error_file()
# Check CMakePresets.json for options