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
set(TESTS_DIR ${CMAKE_SOURCE_DIR}/tests)
set(LOGS_DIR ${CMAKE_SOURCE_DIR}/logs)
set(TESTS_FRAMEWORK_DIR ${TESTS_DIR}/framework)
set(TESTS_DATA_DIR ${TESTS_DIR}/data)
set(CMAKE_DIR ${CMAKE_SOURCE_DIR}/cmake)
set(DEMO_DIR ${CMAKE_SOURCE_DIR}/demo)
set(DEMO_CPP_DIR ${DEMO_DIR})
set(BENCHMARK_DIR ${CMAKE_SOURCE_DIR}/benchmarks)

# ---- Projects Source Directories ----
set(SRC_SHARED_DIR ${SRC_DIR}/shared)
set(SRC_CPP_DIR ${SRC_DIR}/cpp)
set(SRC_CUDA_OPENCV_DIR ${SRC_DIR}/cuda_opencv)
set(SRC_PURE_CUDA_DIR ${SRC_DIR}/pure_cuda)


# ---- Projects Tests Source Directories ----
set(TESTS_CPP_DIR ${TESTS_DIR}/cpp)
set(TESTS_CUDA_OPENCV_DIR ${TESTS_DIR}/cuda_opencv)
set(TESTS_PURE_CUDA_DIR ${TESTS_DIR}/pure_cuda)


# ---- Project Configuration ----
## Set Logic Source for Tests
set(TEST_LOGIC_SRC ${TESTS_FRAMEWORK_DIR}/test_logic.cpp)
## Set Test Includes
set(TEST_LOGIC_INCLUDES ${TESTS_FRAMEWORK_DIR})

# ---- Projects Tests Dependencies ----
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
set(PURE_CUDA_TEST_OUTPUT_DIRECTORIES ${ALL_TESTS_OUTPUT_DIRECTORIS})

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

# ---- Projects Debug Sources ----
## CPU Debug Source
set(CPU_DEBUG_SRC ${SRC_DIR}/debug.cpp)
set(DEBUG_DEFINES_CPU ${DEBUG_DEFINES})
## CUDA OpenCV Debug Source
set(CUDA_OPENCV_DEBUG_SRC ${SRC_DIR}/debug.cpp ${SRC_DIR}/debug.cu)
set(DEBUG_DEFINES_CUDA_OPENCV ${DEBUG_DEFINES})
## PURE CUDA Debug Source
set(PURE_CUDA_DEBUG_SRC ${SRC_DIR}/debug.cpp ${SRC_DIR}/debug.cu)
set(DEBUG_DEFINES_PURE_CUDA ${DEBUG_DEFINES})


# ---- Projects Compilation ----
#set(CPU_INCLUDE ${CMAKE_SOURCE_DIR}/include) ## set as include_directories(${CMAKE_SOURCE_DIR}/include) for all global includes
set(CPU_LINK_LIBS ${OpenCV_LIBS})
#set(CUDA_OPENCV_INCLUDES ${CMAKE_SOURCE_DIR}/include) ## set as include_directories(${CMAKE_SOURCE_DIR}/include) for all global includes
set(CUDA_OPENCV_LINK_LIBS ${OpenCV_LIBS} opencv_cudaimgproc cuda)
#set(PURE_CUDA_INCLUDES ${CMAKE_SOURCE_DIR}/include) ## set as include_directories(${CMAKE_SOURCE_DIR}/include) for all global includes
set(PURE_CUDA_LINK_LIBS ${CUDA_OPENCV_LINK_LIBS})
set(BENCHMARK_LINK_LIBS ${CUDA_OPENCV_LINK_LIBS})

# ---- Projects Logs Settings ----
set(LOG_FILE_PATH "${LOGS_DIR}/cmake_logs.log" CACHE INTERNAL "")
set(ERROR_FILE_PATH "${LOGS_DIR}/cmake_errors.log" CACHE INTERNAL "")


# ---- Projects Utilities Libraries Sources ----
## Shared Utilities Sources
set(SHARED_UTILS_CPP_SRC ${SRC_SHARED_DIR}/shared_utils.cpp)
set(SHARED_UTILS_CUDA_OPENCV_SRC ${SRC_SHARED_DIR}/shared_utils.cpp ${SRC_SHARED_DIR}/shared_utils.cu)
set(SHARED_UTILS_PURE_CUDA_SRC ${SHARED_UTILS_CUDA_OPENCV_SRC})
set(SHARED_UTILS_BENCHMARK_SRC ${SHARED_UTILS_CUDA_OPENCV_SRC})
## Projects Utilities Sources
set(UTILS_CPP_SRC ${SRC_CPP_DIR}/utils_cpp.cpp)
set(UTILS_CUDA_OPENCV_SRC ${SRC_CUDA_OPENCV_DIR}/utils_cuda_opencv.cu)
set(UTILS_PURE_CUDA_SRC ${SRC_PURE_CUDA_DIR}/utils_pure_cuda.cu)
set(BENCHMARK_ALL_UTILS_SRC ${UTILS_CPP_SRC} ${UTILS_CUDA_OPENCV_SRC} ${UTILS_PURE_CUDA_SRC})  # `


# ---- Projects Compilation Sources ----
## C++ Code Files
set(CPU_SRC ${SRC_CPP_DIR}/shape_detector_cpp.cpp ${UTILS_CPP_SRC} ${SHARED_UTILS_CPP_SRC})
## CUDA OpenCV Code Files
set(CUDA_OPENCV_SRC ${SRC_CUDA_OPENCV_DIR}/shape_detector_cuda_opencv.cu ${UTILS_CUDA_OPENCV_SRC} ${SHARED_UTILS_CUDA_OPENCV_SRC})
## PURE CUDA Code Files
set(PURE_CUDA_SRC ${SRC_PURE_CUDA_DIR}/shape_detector_pure_cuda.cu ${UTILS_PURE_CUDA_SRC} ${SHARED_UTILS_PURE_CUDA_SRC})

set(BENCHMARK_SRC ${BENCHMARK_DIR}/utils.cu ${BENCHMARK_ALL_UTILS_SRC} ${SHARED_UTILS_BENCHMARK_SRC}) #


# ---- Executables ----
## C++ DEMO Code Files
set(APP_LOGIC_SRC ${SRC_SHARED_DIR}/shape_detector_logic.cpp)
set(APP_LOGIC_INCLUDES ${SRC_SHARED_DIR})

## Log the start of the configuration
include(${CMAKE_DIR}/AddLogMessage.cmake)
reset_log_file()
#reset_error_file()


# ---- Projects Confinguration Log print ----
include(${CMAKE_DIR}/Logs_ProjectConfig.cmake)











