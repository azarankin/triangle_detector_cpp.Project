# CMakeLists.txt
# This file is the main CMake configuration for the Triangle Detector project.

add_subdirectory(tests)

# ---- Include CMake Modules ----
# Include custom CMake modules for project configuration, logging, and target management
include(${CMAKE_SOURCE_DIR}/cmake/ProjectConfig.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/AddCustomTarget.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/AddGTestExec.cmake)


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

## C++ TEST Code FILES
add_custom_target(triangle_detector_cpp_for_test
    TYPE LIBRARY
    SOURCES ${CPU_SRC} ${CPU_DEBUG_SRC}
    LINK_LIBS ${CPU_LINK_LIBS}
    DEFINES ${DEBUG_DEFINES_CPU}
)
## CPU Tests
add_gtest_exec(
  triangle_detector_cpp_tests
  SOURCES ${TEST_LOGIC_SRC} tests/triangle_cpp_test.cpp
  LIBS_TO_LINK triangle_detector_cpp_for_test
  OUTPUT_DIRS ${CPU_TEST_OUTPUT_DIRECTORIES}
)
add_gtest_exec(
  utils_cpp_tests
  SOURCES ${TEST_LOGIC_SRC} tests/utils_cpp_test.cpp
  LIBS_TO_LINK triangle_detector_cpp_for_test
  OUTPUT_DIRS ${CPU_TEST_OUTPUT_DIRECTORIES}
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
## CUDA Tests
add_gtest_exec(
  triangle_detector_cuda_opencv_tests
  SOURCES ${TEST_LOGIC_SRC} tests/triangle_cuda_opencv_test.cu
  LIBS_TO_LINK triangle_detector_cuda_opencv_for_test
  OUTPUT_DIRS ${CUDA_OPENCV_TEST_OUTPUT_DIRECTORIES}
)
add_gtest_exec(
  utils_cuda_opencv_tests
  SOURCES ${TEST_LOGIC_SRC} tests/utils_cuda_opencv_test.cu
  LIBS_TO_LINK triangle_detector_cuda_opencv_for_test
  OUTPUT_DIRS ${CUDA_OPENCV_TEST_OUTPUT_DIRECTORIES}
)




#---- Executables ----
## CPU Demo
add_custom_target(triangle_detector_cpp_demo
  TYPE EXECUTABLE
  SOURCES demo/demo_cpp.cpp
  LINK_LIBS triangle_detector_cpu_logic
  OUTPUT_DIR "${CMAKE_SOURCE_DIR}/demo/out"
)



set(ALL_TESTS_DEPENDENCIES 
    triangle_detector_cpp_tests
    utils_cpp_tests
    triangle_detector_cuda_opencv_tests
    utils_cuda_opencv_tests
  )

add_gtest_exec(
  run_all_tests
  SOURCES tests/run_all_tests.cpp
  DEPENDENCIES ${ALL_TESTS_DEPENDENCIES}
  OUTPUT_DIRS ${ALL_TESTS_OUTPUT_DIRECTORIES}
)


# also set the ALL_TESTS_DEPENDENCIES inside the tests/run_all_tests.cpp


# ---- Profiling ----


# ---- Banchmarks ----



