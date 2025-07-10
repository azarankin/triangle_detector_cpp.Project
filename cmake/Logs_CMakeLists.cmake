
# ---- Log Project Configuration ----
include(${CMAKE_SOURCE_DIR}/cmake/AddLogMessage.cmake) # Log the start of the configuration

log_message("# ---- CMake Configuration ----
cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})
project(${PROJECT_NAME} VERSION ${PROJECT_VERSION} LANGUAGES ${PROJECT_LANGUAGES})
# ---- Global Configuration ----
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
## Dependencies
find_package(OpenCV REQUIRED)
add_subdirectory(third_party/googletest)
enable_testing()
include(GoogleTest)
")
