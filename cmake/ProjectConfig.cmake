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



# ---- Projects Directions ----
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(SRC_CPP_DIR ${CMAKE_SOURCE_DIR}/src/cpp)
set(SRC_CUDA_OPENCV_DIR ${CMAKE_SOURCE_DIR}/src/cuda_opencv)
set(SRC_SHARED_DIR ${CMAKE_SOURCE_DIR}/src/shared)
set(LOGS_DIR ${CMAKE_SOURCE_DIR}/logs)
set(TESTS_DIR ${CMAKE_SOURCE_DIR}/tests)
set(TESTS_CPP_DIR ${CMAKE_SOURCE_DIR}/tests/cpp)
set(TESTS_CUDA_OPENCV_DIR ${CMAKE_SOURCE_DIR}/tests/cuda_opencv)
set(TESTS_FRAMEWORK_DIR ${CMAKE_SOURCE_DIR}/tests/framework)
set(TESTS_DATA_DIR ${CMAKE_SOURCE_DIR}/tests/data)
set(CMAKE_DIR ${CMAKE_SOURCE_DIR}/cmake)
set(DEMO_DIR ${CMAKE_SOURCE_DIR}/demo)
set(DEMO_CPP_DIR ${DEMO_DIR})

# ---- Project Configuration ----
## Set Logic Source for Tests
set(TEST_LOGIC_SRC ${TESTS_FRAMEWORK_DIR}/test_logic.cpp)
## Set Test Includes
set(TEST_INCLUDES ${TESTS_FRAMEWORK_DIR})
## Test output paths
set(TEST_OUTPUT_DIRECTORY "${TESTS_DATA_DIR}/current_output")
set(TEST_DEBUG_OUTPUT_DIRECTORY "${TEST_OUTPUT_DIRECTORY}/test_debug_output")
set(TEST_OUTPUT_ARCHIVE_DIRECTORY "${TESTS_DATA_DIR}/archive_current_output")

# ---- Tests Output Directories ----
## Set output directories for tests
## The output directories are used for tests results, debug images, and archive of current output
## The output directories will move to archive after each run of tests
## This directories will be cleared for each build
set(ALL_TESTS_OUTPUT_DIRECTORIS ${TEST_OUTPUT_DIRECTORY} ${TEST_DEBUG_OUTPUT_DIRECTORY} ${TEST_OUTPUT_ARCHIVE_DIRECTORY})
set(CPU_TEST_OUTPUT_DIRECTORIES ${ALL_TESTS_OUTPUT_DIRECTORIS})
set(CUDA_OPENCV_TEST_OUTPUT_DIRECTORIES ${ALL_TESTS_OUTPUT_DIRECTORIS})
set(ALL_TESTS_OUTPUT_DIRECTORIES ${ALL_TESTS_OUTPUT_DIRECTORIS}) ## additional output directory list

## Test reference paths
set(TEST_REFERENCE_INPUT_DIRECTORY "${TESTS_DATA_DIR}/reference_input")
set(TEST_REFERENCE_OUTPUT_DIRECTORY "${TESTS_DATA_DIR}/reference_output")


## Debug Targets
set(DEBUG_DEFINES
    ENABLE_DEBUG_IMAGES ON
    DEBUG_IMAGE_DIR="${TEST_OUTPUT_DIRECTORY}/debug"
    TEST_OUTPUT_DIRECTORY="${TEST_OUTPUT_DIRECTORY}"
    TEST_DEBUG_OUTPUT_DIRECTORY="${TEST_DEBUG_OUTPUT_DIRECTORY}"
    TEST_OUTPUT_ARCHIVE_DIRECTORY="${TEST_OUTPUT_ARCHIVE_DIRECTORY}"
    TEST_REFERENCE_INPUT_DIRECTORY="${TEST_REFERENCE_INPUT_DIRECTORY}"
    TEST_REFERENCE_OUTPUT_DIRECTORY="${TEST_REFERENCE_OUTPUT_DIRECTORY}"
)

set(CPU_DEBUG_SRC ${SRC_DIR}/debug.cpp)
set(DEBUG_DEFINES_CPU ${DEBUG_DEFINES})

set(CUDA_OPENCV_DEBUG_SRC ${SRC_DIR}/debug.cpp ${SRC_DIR}/debug.cu)
set(DEBUG_DEFINES_CUDA_OPENCV ${DEBUG_DEFINES})


# ---- Projects Compilation ----
#set(CPU_INCLUDE ${CMAKE_SOURCE_DIR}/include) ## set as include_directories(${CMAKE_SOURCE_DIR}/include) for all global includes
set(CPU_LINK_LIBS ${OpenCV_LIBS})
#set(CUDA_OPENCV_INCLUDES ${CMAKE_SOURCE_DIR}/include)
set(CUDA_OPENCV_LINK_LIBS ${OpenCV_LIBS} opencv_cudaimgproc cuda)

# ---- Projects Logs Settings ----
set(LOG_FILE_PATH "${LOGS_DIR}/cmake_logs.log" CACHE INTERNAL "")
set(ERROR_FILE_PATH "${LOGS_DIR}/cmake_errors.log" CACHE INTERNAL "")



# ---- Projects Compilation ----
## C++ Code Files
set(CPU_SRC ${SRC_CPP_DIR}/shape_detector_cpp.cpp ${SRC_SHARED_DIR}/shared_utils.cpp ${SRC_CPP_DIR}/utils_cpp.cpp)
## CUDA OpenCV Code Files
set(CUDA_OPENCV_SRC ${SRC_CUDA_OPENCV_DIR}/shape_detector_cuda_opencv.cu ${SRC_SHARED_DIR}/shared_utils.cpp ${SRC_SHARED_DIR}/shared_utils.cu ${SRC_CUDA_OPENCV_DIR}/utils_cuda_opencv.cu)
##
set(APP_LOGIC_SRC ${SRC_SHARED_DIR}/shape_detector_logic.cpp)
set(APP_LOGIC_INCLUDES ${SRC_SHARED_DIR})

## Log the start of the configuration
include(${CMAKE_DIR}/AddLogMessage.cmake)
reset_log_file()
#reset_error_file()


# ---- Projects Confinguration Log print ----
include(${CMAKE_DIR}/Logs_ProjectConfig.cmake)











