# tests/CMakeLists.txt

option(ENABLE_ALL_TESTS_COMPILATION "..." OFF)
option(ENABLE_CPP_TESTS "..." OFF)
option(ENABLE_CUDA_OPENCV_TESTS "..." OFF)
option(ENABLE_PURE_CUDA_TESTS "..." OFF)



include(${CMAKE_DIR}/AddCustomTarget.cmake)
include(${CMAKE_DIR}/AddGTestExec.cmake)



if(ENABLE_CPP_TESTS)

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
    INCLUDES ${TEST_LOGIC_INCLUDES}
    )
    add_gtest_exec(
    utils_cpp_tests
    SOURCES ${TEST_LOGIC_SRC} ${TESTS_CPP_DIR}/utils_cpp_test.cpp
    LIBS_TO_LINK triangle_detector_cpp_for_test
    OUTPUT_DIRS ${CPU_TEST_OUTPUT_DIRECTORIES}
    INCLUDES ${TEST_LOGIC_INCLUDES}
    )

endif()



if(ENABLE_CUDA_OPENCV_TESTS)

    # CUDA OpenCV TEST Code FILES
    add_custom_target(triangle_detector_cuda_opencv_for_test
        TYPE LIBRARY
        SOURCES ${CUDA_OPENCV_SRC} ${CUDA_OPENCV_DEBUG_SRC}
        LINK_LIBS ${CUDA_OPENCV_LINK_LIBS}
        DEFINES ${DEBUG_DEFINES_CUDA_OPENCV}
    )

    ## CUDA OPENCV Tests
    add_gtest_exec(
    triangle_detector_cuda_opencv_tests
    SOURCES ${TEST_LOGIC_SRC} ${TESTS_CUDA_OPENCV_DIR}/triangle_cuda_opencv_test.cu
    LIBS_TO_LINK triangle_detector_cuda_opencv_for_test
    OUTPUT_DIRS ${CUDA_OPENCV_TEST_OUTPUT_DIRECTORIES}
    INCLUDES ${TEST_LOGIC_INCLUDES}
    )
    add_gtest_exec(
    utils_cuda_opencv_tests
    SOURCES ${TEST_LOGIC_SRC} ${TESTS_CUDA_OPENCV_DIR}/utils_cuda_opencv_test.cu
    LIBS_TO_LINK triangle_detector_cuda_opencv_for_test
    OUTPUT_DIRS ${CUDA_OPENCV_TEST_OUTPUT_DIRECTORIES}
    INCLUDES ${TEST_LOGIC_INCLUDES}
    )


endif()



if(ENABLE_PURE_CUDA_TESTS)

    # CUDA TEST Code FILES
    add_custom_target(triangle_detector_pure_cuda_for_test
        TYPE LIBRARY
        SOURCES ${PURE_CUDA_SRC} ${PURE_CUDA_DEBUG_SRC}
        LINK_LIBS ${PURE_CUDA_LINK_LIBS}
        DEFINES ${DEBUG_DEFINES_PURE_CUDA}
    )

    ## CUDA Tests
    add_gtest_exec(
    triangle_detector_pure_cuda_tests
    SOURCES ${TEST_LOGIC_SRC} ${TESTS_PURE_CUDA_DIR}/triangle_pure_cuda_test.cu
    LIBS_TO_LINK triangle_detector_pure_cuda_for_test
    OUTPUT_DIRS ${PURE_CUDA_TEST_OUTPUT_DIRECTORIES}
    INCLUDES ${TEST_LOGIC_INCLUDES}
    )
    add_gtest_exec(
    utils_pure_cuda_tests
    SOURCES ${TEST_LOGIC_SRC} ${TESTS_PURE_CUDA_DIR}/utils_pure_cuda_test.cu
    LIBS_TO_LINK triangle_detector_pure_cuda_for_test
    OUTPUT_DIRS ${PURE_CUDA_TEST_OUTPUT_DIRECTORIES}
    INCLUDES ${TEST_LOGIC_INCLUDES}
    )

endif()



# ---- Define several tests runner ----
# This is the main test runner that will run all tests defined above.
# It will be used to run all tests in one go, and it will depend on all
# the individual test targets defined above.
# Define the output directories for all tests

if(ENABLE_ALL_TESTS_COMPILATION)

    set(ALL_TRIANGLE_DETECTOR_TESTS_DEPENDENCIES 
        triangle_detector_cpp_tests
        triangle_detector_cuda_opencv_tests
    )

    add_gtest_exec(
    triangle_detector_all_tests
    SOURCES ${TESTS_DIR}/run_all_tests.cpp
    DEPENDENCIES ${ALL_TRIANGLE_DETECTOR_TESTS_DEPENDENCIES}
    )


    set(ALL_UTILS_TESTS_DEPENDENCIES 
        utils_cpp_tests
        utils_cuda_opencv_tests
    )

    add_gtest_exec(
    utils_all_tests
    SOURCES ${TESTS_DIR}/run_all_tests.cpp
    DEPENDENCIES ${ALL_UTILS_TESTS_DEPENDENCIES}
    )


    set(ALL_TESTS_DEPENDENCIES 
        ${ALL_TRIANGLE_DETECTOR_TESTS_DEPENDENCIES}
        ${ALL_UTILS_TESTS_DEPENDENCIES}
    )

    add_gtest_exec(
    run_all_tests
    SOURCES ${TESTS_DIR}/run_all_tests.cpp
    DEPENDENCIES ${ALL_TESTS_DEPENDENCIES}
    )


endif()











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