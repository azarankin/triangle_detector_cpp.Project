# tests/CMakeLists.txt


include(${CMAKE_DIR}/AddCustomTarget.cmake)
include(${CMAKE_DIR}/AddGTestExec.cmake)

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
  SOURCES ${TEST_LOGIC_SRC} ${TESTS_CPP_DIR}/triangle_cpp_test.cpp
  LIBS_TO_LINK triangle_detector_cpp_for_test
  OUTPUT_DIRS ${CPU_TEST_OUTPUT_DIRECTORIES}
  INCLUDES ${TEST_INCLUDES}
)
add_gtest_exec(
  utils_cpp_tests
  SOURCES ${TEST_LOGIC_SRC} ${TESTS_CPP_DIR}/utils_cpp_test.cpp
  LIBS_TO_LINK triangle_detector_cpp_for_test
  OUTPUT_DIRS ${CPU_TEST_OUTPUT_DIRECTORIES}
  INCLUDES ${TEST_INCLUDES}
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
  SOURCES ${TEST_LOGIC_SRC} ${TESTS_CUDA_OPENCV_DIR}/triangle_cuda_opencv_test.cu
  LIBS_TO_LINK triangle_detector_cuda_opencv_for_test
  OUTPUT_DIRS ${CUDA_OPENCV_TEST_OUTPUT_DIRECTORIES}
  INCLUDES ${TEST_INCLUDES}
)
add_gtest_exec(
  utils_cuda_opencv_tests
  SOURCES ${TEST_LOGIC_SRC} ${TESTS_CUDA_OPENCV_DIR}/utils_cuda_opencv_test.cu
  LIBS_TO_LINK triangle_detector_cuda_opencv_for_test
  OUTPUT_DIRS ${CUDA_OPENCV_TEST_OUTPUT_DIRECTORIES}
  INCLUDES ${TEST_INCLUDES}
)


# Define the output directories for all tests

set(ALL_TESTS_DEPENDENCIES 
    triangle_detector_cpp_tests
    utils_cpp_tests
    triangle_detector_cuda_opencv_tests
    utils_cuda_opencv_tests
  )

add_gtest_exec(
  run_all_tests
  SOURCES ${TESTS_DIR}/run_all_tests.cpp
  DEPENDENCIES ${ALL_TESTS_DEPENDENCIES}
  OUTPUT_DIRS ${ALL_TESTS_OUTPUT_DIRECTORIES}
)








# להגדיר קבצי הטסטים שלך
# set(TEST_SOURCES
#     triangle_cpp_test.cpp
#     utils_cpp_test.cpp
#     ...
# )

# add_executable(run_all_tests ${TEST_SOURCES})
# # לקשר לספריות שאתה בודק (שנבנות בפרויקט הראשי)
# target_link_libraries(run_all_tests PRIVATE triangle_detector_cpp_for_test ${OpenCV_LIBS} gtest gtest_main)

# # תוספות אופציונליות
# gtest_discover_tests(run_all_tests)