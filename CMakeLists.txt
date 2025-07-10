
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
add_subdirectory(third_party/googletest)
enable_testing()
include(GoogleTest)

## Log the start of the configuration
include(${CMAKE_SOURCE_DIR}/cmake/AddLogMessage.cmake)
reset_log_file()
reset_error_file()


# ---- Include CMake Modules ----
# Include custom CMake modules for project configuration, logging, and target management
include(${CMAKE_SOURCE_DIR}/cmake/ProjectConfig.cmake)


include(${CMAKE_SOURCE_DIR}/cmake/Logs_CMakeLists.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/AddCustomTarget.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/AddGTestExec.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/AddPostBuildClearOutput.cmake)



# ---- Projects Compilation ----
## C++ Code Files
set(CPU_SRC src/cpp/shape_detector_cpp.cpp src/shared/shared_utils.cpp src/cpp/utils_cpp.cpp)
## CUDA OpenCV Code Files
set(CUDA_OPENCV_SRC src/cuda_opencv/shape_detector_cuda_opencv.cu src/shared/shared_utils.cpp src/shared/shared_utils.cu src/cuda_opencv/utils_cuda_opencv.cu)



## C++ Code Files
add_custom_target(triangle_detector_cpu
    TYPE LIBRARY
    SOURCES ${CPU_SRC}
    LINK_LIBS ${CPU_LINK_LIBS}
)

## C++ DEMO Code Files
add_custom_target(triangle_detector_cpu_logic
    TYPE LIBRARY
    SOURCES ${CPU_SRC} src/shared/shape_detector_logic.cpp
    LINK_LIBS ${CPU_LINK_LIBS}
)

# C++ TEST Code FILES
add_custom_target(triangle_detector_cpp_for_test
    TYPE LIBRARY
    SOURCES ${CPU_SRC} ${CPU_DEBUG_SRC}
    LINK_LIBS ${CPU_LINK_LIBS}
    DEFINES ${DEBUG_DEFINES_CPU}
)


## CUDA OpenCV Code Files
add_custom_target(triangle_detector_cuda_opencv
    TYPE LIBRARY
    SOURCES ${CUDA_OPENCV_SRC}
    LINK_LIBS ${CUDA_OPENCV_LINK_LIBS}
)

# CUDA OpenCV TEST Code FILES
add_custom_target(triangle_detector_cuda_opencv_for_test
    TYPE LIBRARY
    SOURCES ${CUDA_OPENCV_SRC} ${CUDA_OPENCV_DEBUG_SRC}
    LINK_LIBS ${CUDA_OPENCV_LINK_LIBS}
    DEFINES ${DEBUG_DEFINES_CUDA_OPENCV}
)

#---- Executables ----
## CPU Demo
add_custom_target(triangle_detector_cpp_demo
  TYPE EXECUTABLE
  SOURCES demo/demo_cpp.cpp
  LINK_LIBS triangle_detector_cpu_logic
  OUTPUT_DIR "${CMAKE_SOURCE_DIR}/demo/out"
)


## CPU Tests
add_gtest_exec(
  triangle_detector_cpp_tests
  SOURCES ${TEST_LOGIC_SRC} tests/triangle_cpp_test.cpp tests/utils_cpp_test.cpp
  LIBS_TO_LINK triangle_detector_cpp_for_test
  OUTPUT_DIRS ${CPU_TEST_OUTPUT_DIRECTORIES}
)


## CUDA Tests
add_gtest_exec(
  triangle_detector_cuda_opencv_tests
  SOURCES ${TEST_LOGIC_SRC} tests/triangle_cuda_opencv_test.cpp
  LIBS_TO_LINK triangle_detector_cuda_opencv_for_test
  OUTPUT_DIRS ${CUDA_OPENCV_TEST_OUTPUT_DIRECTORIES}
)



set(ALL_TESTS_DEPENDENCIES triangle_detector_cpp_tests triangle_detector_cuda_opencv_tests)

add_gtest_exec(
  run_all_tests
  SOURCES tests/run_all_tests.cpp
  DEPENDENCIES ${ALL_TESTS_DEPENDENCIES}
  OUTPUT_DIRS ${ALL_TESTS_OUTPUT_DIRECTORIES}
)


# also set the ALL_TESTS_DEPENDENCIES inside the tests/run_all_tests.cpp


# ---- Profiling ----


# ---- Banchmarks ----



