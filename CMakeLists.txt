# CMakeLists.txt
# This file is the main CMake configuration for the Triangle Detector project.

# ---- CMake Configuration ----
set(CMAKE_MINIMUM_REQUIRED_VERSION 3.19)
cmake_minimum_required(VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

# ---- Project Configuration ----
set(PROJECT_NAME triangle_detector_project)
set(PROJECT_VERSION 1.0.0)
set(PROJECT_LANGUAGES CXX CUDA)
project(${PROJECT_NAME} VERSION ${PROJECT_VERSION} LANGUAGES ${PROJECT_LANGUAGES})


option(ENABLE_EXECUTABLES "..." OFF)



# ---- Include CMake Modules ----
# Include custom CMake modules for project configuration, logging, and target management
include(${CMAKE_SOURCE_DIR}/cmake/ProjectConfig.cmake)
include(${CMAKE_DIR}/AddCustomTarget.cmake)

## C++ Code Files
add_custom_target(triangle_detector_cpu
    TYPE LIBRARY
    SOURCES ${CPU_SRC}
    LINK_LIBS ${CPU_LINK_LIBS}
)


## CUDA OpenCV Code Files
add_custom_target(triangle_detector_cuda_opencv
    TYPE LIBRARY
    SOURCES ${CUDA_OPENCV_SRC}
    LINK_LIBS ${CUDA_OPENCV_LINK_LIBS}
)


## CUDA Code Files
add_custom_target(triangle_detector_pure_cuda
    TYPE LIBRARY
    SOURCES ${PURE_CUDA_SRC}
    LINK_LIBS ${PURE_CUDA_LINK_LIBS}
)


# POC Benchmark Code Files

add_custom_target(utils_benchmark
    TYPE EXECUTABLE
    SOURCES ${BENCHMARK_DIR}/main.cu ${BENCHMARK_SRC}
    LINK_LIBS ${BENCHMARK_LINK_LIBS}
    OUTPUT_DIR "${BENCHMARK_DIR}/out"
    INCLUDES ${BENCHMARK_DIR}
)




if(ENABLE_EXECUTABLES)
    #---- Executables ----
    ## C++ DEMO Code Files
    add_custom_target(triangle_detector_cpu_logic
        TYPE LIBRARY
        SOURCES ${CPU_SRC} ${APP_LOGIC_SRC}
        LINK_LIBS ${CPU_LINK_LIBS}
        INCLUDES ${APP_LOGIC_INCLUDES}
    )
    ## CPU Demo
    add_custom_target(triangle_detector_cpp_demo
    TYPE EXECUTABLE
    SOURCES ${DEMO_CPP_DIR}/demo_cpp.cpp
    LINK_LIBS triangle_detector_cpu_logic
    OUTPUT_DIR "${DEMO_CPP_DIR}/out"
    )


endif()




# ---- Tests ----
add_subdirectory(${CMAKE_SOURCE_DIR}/tests)






# also set the ALL_TESTS_DEPENDENCIES inside the tests/run_all_tests.cpp


# ---- Profiling ----


# ---- Banchmarks ----



