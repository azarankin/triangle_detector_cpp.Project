
# ---- Log Project Configuration ----
include(${CMAKE_SOURCE_DIR}/cmake/AddLogMessage.cmake) # Log the start of the configuration

log_message("---- CMake Configuration ----
cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})
project(${PROJECT_NAME} VERSION ${PROJECT_VERSION} LANGUAGES ${PROJECT_LANGUAGES})
---- Global Configuration ----
CMAKE_BUILD_TYPE = \"${CMAKE_BUILD_TYPE}\"
enable_language(CUDA)
CMAKE_CXX_STANDARD_REQUIRED = \"${CMAKE_CXX_STANDARD_REQUIRED}\"
CMAKE_CUDA_STANDARD_REQUIRED = \"${CMAKE_CUDA_STANDARD_REQUIRED}\"
CMAKE_CXX_STANDARD = \"${CMAKE_CXX_STANDARD}\"
CMAKE_CUDA_STANDARD = \"${CMAKE_CUDA_STANDARD}\"
CMAKE_CUDA_ARCHITECTURES = \"${CMAKE_CUDA_ARCHITECTURES}\"  ## Set inside CMakePresets.json
CMAKE_CUDA_COMPILER = \"${CMAKE_CUDA_COMPILER}\" # CUDA Path ## Set inside CMakePresets.json
OpenCV_INCLUDE_DIRS = \"${OpenCV_INCLUDE_DIRS}\"
OpenCV_LIBS = \"${OpenCV_LIBS}\"
CMAKE_CUDA_FLAGS = \"${CMAKE_CUDA_FLAGS}\"
CMAKE_POSITION_INDEPENDENT_CODE = \"${CMAKE_POSITION_INDEPENDENT_CODE}\" ## Library build
add_compile_options(${COMPILER_WARNINGS}) ## Show errors on production
include_directories(${INCLUDE_PROJECT_DIRECTORIES}) ## Project Include Directories
add_compile_definitions(DEBUG_IMAGE_DEFAULT_DIR=\"${DEBUG_IMAGE_DEFAULT_DIR}\") ## Project Debug Default Configuration
Dependencies
find_package(OpenCV REQUIRED)
add_subdirectory(third_party/googletest)
enable_testing()
include(GoogleTest)

---- Project Configuration ----
Set Logic Source for Tests
TEST_LOGIC_SRC = \"${TEST_LOGIC_SRC}\"
Test output paths
TEST_OUTPUT_DIRECTORY = \"${TEST_OUTPUT_DIRECTORY}\"
TEST_OUTPUT_ARCHIVE_DIRECTORY = \"${TEST_OUTPUT_ARCHIVE_DIRECTORY}\"
CPU_TEST_OUTPUT_DIRECTORIES = \"${CPU_TEST_OUTPUT_DIRECTORIES}\"
CUDA_OPENCV_TEST_OUTPUT_DIRECTORIES = \"${CUDA_OPENCV_TEST_OUTPUT_DIRECTORIES}\"
ALL_TESTS_OUTPUT_DIRECTORIES = \"${ALL_TESTS_OUTPUT_DIRECTORIES}\"  ## additional output directory list
Debug Targets
DEBUG_DEFINES = \"${DEBUG_DEFINES}\"
CPU_DEBUG_SRC = \"${CPU_DEBUG_SRC}\"
DEBUG_DEFINES_CPU = \"${DEBUG_DEFINES_CPU}\"
DEBUG_DEFINES_CUDA_OPENCV = \"${DEBUG_DEFINES_CUDA_OPENCV}\"
CUDA_OPENCV_DEBUG_SRC = \"${CUDA_OPENCV_DEBUG_SRC}\"
---- Projects Compilation ----
CPU_LINK_LIBS = \"${CPU_LINK_LIBS}\"
CUDA_OPENCV_LINK_LIBS = \"${CUDA_OPENCV_LINK_LIBS}\"
---- Projects Logs ----
LOG_FILE_PATH = \"${LOG_FILE_PATH}
ERROR_FILE_PATH = \"${ERROR_FILE_PATH}\"
")
