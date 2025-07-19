# tests/CMakeLists.txt


# Check CMakePresets.json for options

set(TEST_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/current_output")
set(TEST_OUTPUT_ARCHIVE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/archive_current_output")
set(TEST_REFERENCE_INPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/reference_input")
set(TEST_REFERENCE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/reference_output")

# Set DEBUG_DEFINES ON inside CMakePresets.json if needed




set(TEST_DEFINES
    
    TEST_OUTPUT_DIRECTORY="${TEST_OUTPUT_DIRECTORY}"
    TEST_DEBUG_OUTPUT_DIRECTORY="${TEST_OUTPUT_DIRECTORY}/test_debug_output"
    TEST_OUTPUT_ARCHIVE_DIRECTORY="${TEST_OUTPUT_ARCHIVE_DIRECTORY}"
    TEST_REFERENCE_INPUT_DIRECTORY="${TEST_REFERENCE_INPUT_DIRECTORY}"
    TEST_REFERENCE_OUTPUT_DIRECTORY="${TEST_REFERENCE_OUTPUT_DIRECTORY}"
)


add_library(test_logic_lib STATIC ${CMAKE_CURRENT_SOURCE_DIR}/framework/test_logic.cpp)
target_link_libraries(test_logic_lib PUBLIC gtest gtest_main OpenCV_Interface)
target_include_directories(test_logic_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/framework)
target_compile_definitions(test_logic_lib PUBLIC ${TEST_DEFINES})
add_postbuild_clear_output(test_logic_lib ${TEST_OUTPUT_DIRECTORY})
add_postbuild_clear_output(test_logic_lib ${TEST_OUTPUT_ARCHIVE_DIRECTORY})


add_library(TestInterface INTERFACE)
target_link_libraries(TestInterface INTERFACE test_logic_lib)





if(ENABLE_CPP_TESTS OR SWITCH_TO_ALL_TESTS_COMPILATION)
    #OUTPUT_DIRS ${CPU_TEST_OUTPUT_DIRECTORIES}

    ## CPU Tests
    add_executable(triangle_detector_cpp_tests cpp/triangle_cpp_test.cpp)
    target_link_libraries(triangle_detector_cpp_tests PRIVATE cpp_shape_detector_lib TestInterface)
    if(NOT DEFINED SWITCH_TO_ALL_TESTS_COMPILATION)
        gtest_discover_tests(triangle_detector_cpp_tests)
    endif()
    

    add_executable(utils_cpp_tests cpp/utils_cpp_test.cpp)
    target_link_libraries(utils_cpp_tests PRIVATE cpp_utils_lib TestInterface)
    if(NOT DEFINED SWITCH_TO_ALL_TESTS_COMPILATION)
        gtest_discover_tests(utils_cpp_tests)
    endif()

endif()



if(ENABLE_CUDA_OPENCV_TESTS OR SWITCH_TO_ALL_TESTS_COMPILATION)
    #OUTPUT_DIRS ${CUDA_OPENCV_TEST_OUTPUT_DIRECTORIES}


    ## CUDA OPENCV Tests
    add_executable(triangle_detector_cuda_opencv_tests cuda_opencv/triangle_cuda_opencv_test.cu)
    target_link_libraries(triangle_detector_cuda_opencv_tests PRIVATE cuda_opencv_shape_detector_lib TestInterface)
    if(NOT DEFINED SWITCH_TO_ALL_TESTS_COMPILATION)
        gtest_discover_tests(triangle_detector_cuda_opencv_tests)
    endif()

    add_executable(utils_cuda_opencv_tests cuda_opencv/utils_cuda_opencv_test.cu)
    target_link_libraries(utils_cuda_opencv_tests PRIVATE cuda_opencv_utils_lib TestInterface)
    if(NOT DEFINED SWITCH_TO_ALL_TESTS_COMPILATION)
        gtest_discover_tests(utils_cuda_opencv_tests)
    endif()

endif()



if(ENABLE_PURE_CUDA_TESTS OR SWITCH_TO_ALL_TESTS_COMPILATION)
    #OUTPUT_DIRS ${PURE_CUDA_TEST_OUTPUT_DIRECTORIES}


    # CUDA TESTS

    add_executable(triangle_detector_pure_cuda_tests pure_cuda/triangle_pure_cuda_test.cu)
    target_link_libraries(triangle_detector_pure_cuda_tests PRIVATE pure_cuda_shape_detector_lib TestInterface)
    if(NOT DEFINED SWITCH_TO_ALL_TESTS_COMPILATION)
        gtest_discover_tests(triangle_detector_pure_cuda_tests)
    endif()


    add_executable(utils_pure_cuda_tests pure_cuda/utils_pure_cuda_test.cu)
    target_link_libraries(utils_pure_cuda_tests PRIVATE pure_cuda_utils_lib TestInterface)
    if(NOT DEFINED SWITCH_TO_ALL_TESTS_COMPILATION)
        gtest_discover_tests(utils_pure_cuda_tests)
    endif()

endif()



# ---- Define several tests runner ----
# This is the main test runner that will run all tests defined above.
# It will be used to run all tests in one go, and it will depend on all
# the individual test targets defined above.
# Define the output directories for all tests

if(SWITCH_TO_ALL_TESTS_COMPILATION)

    set(ALL_TRIANGLE_DETECTOR_TESTS_DEPENDENCIES 
        triangle_detector_cpp_tests
        triangle_detector_cuda_opencv_tests
        triangle_detector_pure_cuda_tests
    )

    add_executable(_triangle_detector_all_tests run_all_tests.cpp)
    string(REPLACE ";" "," TESTS_BINARIES "${ALL_TRIANGLE_DETECTOR_TESTS_DEPENDENCIES}")
    target_compile_definitions(_triangle_detector_all_tests PUBLIC ALL_TESTS_BINARIES="${TESTS_BINARIES}")


    set(ALL_UTILS_TESTS_DEPENDENCIES 
        utils_cpp_tests
        utils_cuda_opencv_tests
        utils_pure_cuda_tests
    )


    add_executable(_utils_all_tests run_all_tests.cpp)
    string(REPLACE ";" "," TESTS_BINARIES "${ALL_UTILS_TESTS_DEPENDENCIES}")
    target_compile_definitions(_utils_all_tests PUBLIC ALL_TESTS_BINARIES="${TESTS_BINARIES}")


    set(ALL_TESTS_DEPENDENCIES 
        ${ALL_TRIANGLE_DETECTOR_TESTS_DEPENDENCIES}
        ${ALL_UTILS_TESTS_DEPENDENCIES}
    )

    add_executable(_run_all_tests run_all_tests.cpp)
    string(REPLACE ";" "," TESTS_BINARIES "${ALL_TESTS_DEPENDENCIES}")
    target_compile_definitions(_run_all_tests PUBLIC ALL_TESTS_BINARIES="${TESTS_BINARIES}")



endif()
