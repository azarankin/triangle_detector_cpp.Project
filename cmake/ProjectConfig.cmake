
# ---- CMake Configuration ----
set(CMAKE_MINIMUM_REQUIRED_VERSION 3.19)
cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

# ---- Project Configuration ----
set(PROJECT_NAME triangle_detector_cpp)
set(PROJECT_VERSION 1.0.0)
set(PROJECT_LANGUAGES CXX CUDA)
project(${PROJECT_NAME} VERSION ${PROJECT_VERSION} LANGUAGES ${PROJECT_LANGUAGES})

# ---- Global Configuration ----
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
set(INCLUDE_PROJECT_DIRECTORIES ${CMAKE_SOURCE_DIR}/include)
include_directories(${INCLUDE_PROJECT_DIRECTORIES}) ## Project Include Directories  # For each target  target_include_directories(<target_name> PUBLIC ${CMAKE_SOURCE_DIR}/include)
set(DEBUG_IMAGE_DEFAULT_DIR "${CMAKE_SOURCE_DIR}/debug")
add_compile_definitions(DEBUG_IMAGE_DEFAULT_DIR="${DEBUG_IMAGE_DEFAULT_DIR}") ## Project Debug Default Configuration  #file(MAKE_DIRECTORY "${DEBUG_IMAGE_DEFAULT_DIR}") # create debug directory if it doesn't exist

## Dependencies
find_package(OpenCV REQUIRED)
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/googletest)
enable_testing()
include(GoogleTest)


## Log the start of the configuration
include(${CMAKE_SOURCE_DIR}/cmake/AddLogMessage.cmake)
reset_log_file()
#reset_error_file()



# ---- Project Configuration ----
## Set Logic Source for Tests
set(TEST_LOGIC_SRC tests/framework/test_logic.cpp)
## Test output paths
set(TEST_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tests/data/current_output")
set(TEST_OUTPUT_ARCHIVE_DIRECTORY "${CMAKE_SOURCE_DIR}/tests/data/archive_current_output")
set(CPU_TEST_OUTPUT_DIRECTORIES ${TEST_OUTPUT_DIRECTORY} ${TEST_OUTPUT_ARCHIVE_DIRECTORY})
set(CUDA_OPENCV_TEST_OUTPUT_DIRECTORIES ${TEST_OUTPUT_DIRECTORY} ${TEST_OUTPUT_ARCHIVE_DIRECTORY})
set(ALL_TESTS_OUTPUT_DIRECTORIES ${TEST_OUTPUT_DIRECTORY} ${TEST_OUTPUT_ARCHIVE_DIRECTORY}) ## additional output directory list

## Debug Targets
set(DEBUG_DEFINES
    ENABLE_DEBUG_IMAGES ON
    DEBUG_IMAGE_DIR="${TEST_OUTPUT_DIRECTORY}/debug"
    TEST_OUTPUT_DIRECTORY="${TEST_OUTPUT_DIRECTORY}"
    TEST_OUTPUT_ARCHIVE_DIRECTORY="${TEST_OUTPUT_ARCHIVE_DIRECTORY}"
)
set(CPU_DEBUG_SRC src/debug.cpp)
set(DEBUG_DEFINES_CPU ${DEBUG_DEFINES})
set(DEBUG_DEFINES_CUDA_OPENCV ${DEBUG_DEFINES})
set(CUDA_OPENCV_DEBUG_SRC src/debug.cpp src/debug.cu)

# ---- Projects Compilation ----
#set(CPU_INCLUDE ${CMAKE_SOURCE_DIR}/include) ## set as include_directories(${CMAKE_SOURCE_DIR}/include) for all global includes
set(CPU_LINK_LIBS ${OpenCV_LIBS})
#set(CUDA_OPENCV_INCLUDES ${CMAKE_SOURCE_DIR}/include)
set(CUDA_OPENCV_LINK_LIBS ${OpenCV_LIBS} opencv_cudaimgproc cuda)

# ---- Projects Logs Settings ----
set(LOG_FILE_PATH "${CMAKE_SOURCE_DIR}/logs/cmake_logs.log" CACHE INTERNAL "")
set(ERROR_FILE_PATH "${CMAKE_SOURCE_DIR}/logs/cmake_errors.log" CACHE INTERNAL "")



# ---- Projects Compilation ----
## C++ Code Files
set(CPU_SRC src/cpp/shape_detector_cpp.cpp src/shared/shared_utils.cpp src/cpp/utils_cpp.cpp)
## CUDA OpenCV Code Files
set(CUDA_OPENCV_SRC src/cuda_opencv/shape_detector_cuda_opencv.cu src/shared/shared_utils.cpp src/shared/shared_utils.cu src/cuda_opencv/utils_cuda_opencv.cu)



## Log the start of the configuration
include(${CMAKE_SOURCE_DIR}/cmake/AddLogMessage.cmake)
reset_log_file()
#reset_error_file()


# ---- Projects Confinguration Log print ----
include(${CMAKE_SOURCE_DIR}/cmake/Logs_ProjectConfig.cmake)











