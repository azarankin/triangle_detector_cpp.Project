
cmake_minimum_required(VERSION 3.19) 
message("cmake_minimum_required(VERSION 3.19)")

project(triangle_detector LANGUAGES CXX CUDA)
message("====> project(triangle_detector LANGUAGES CXX CUDA)")

# ---- Global Configuration ----
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

# Set in CMakePresets.json
#set(CMAKE_CUDA_ARCHITECTURES 89)
#set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc) # CUDA Path

message("==> CMAKE_CXX_STANDARD = \"${CMAKE_CXX_STANDARD}\"")
message("==> CMAKE_CUDA_STANDARD = \"${CMAKE_CUDA_STANDARD}\"")
message("==> CMAKE_CUDA_ARCHITECTURES = \"${CMAKE_CUDA_ARCHITECTURES}\"")
message("==> CMAKE_CUDA_COMPILER = \"${CMAKE_CUDA_COMPILER}\"")
message("==> OpenCV_INCLUDE_DIRS = \"${OpenCV_INCLUDE_DIRS}\"")
message("==> OpenCV_LIBS = \"${OpenCV_LIBS}\"")
#set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3 -DNDEBUG")

# Library build
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
message("==> CMAKE_POSITION_INDEPENDENT_CODE = \"${CMAKE_POSITION_INDEPENDENT_CODE}\"")

# ---- Debug Configuration ----
set(DEBUG_IMAGE_DEFAULT_DIR "${CMAKE_SOURCE_DIR}/debug")
add_compile_definitions(
    DEBUG_IMAGE_DEFAULT_DIR="${DEBUG_IMAGE_DEFAULT_DIR}"
)
#file(MAKE_DIRECTORY "${DEBUG_IMAGE_DEFAULT_DIR}") # create debug directory if it doesn't exist


# Test output paths
set(TEST_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tests/current_output")
set(TEST_OUTPUT_ARCHIVE_DIRECTORY "${CMAKE_SOURCE_DIR}/tests/archive_current_output")
message("==> TEST_OUTPUT_DIRECTORY = \"${TEST_OUTPUT_DIRECTORY}\"")
message("==> TEST_OUTPUT_ARCHIVE_DIRECTORY = \"${TEST_OUTPUT_ARCHIVE_DIRECTORY}\"")



# ---- Include Directories ----
include_directories(${CMAKE_SOURCE_DIR}/include)
message("include_directories(${CMAKE_SOURCE_DIR})")
message("include_directories(${CMAKE_SOURCE_DIR}/include)")
# For each target  target_include_directories(<target_name> PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Show errors on production
add_compile_options(-Wall -Wextra -Wno-unused-parameter)
message("add_compile_options(-Wall -Wextra -Wno-unused-parameter)")
# For each target target_compile_options(<target_name> PRIVATE -Wall -Wextra)



# ---- Dependencies ----
find_package(OpenCV REQUIRED)
add_subdirectory(third_party/googletest)
enable_testing()
include(GoogleTest)
message("
find_package(OpenCV REQUIRED)
add_subdirectory(third_party/googletest)
enable_testing()
include(GoogleTest)")



# ---- Utility: Post-build Cleanup ----
function(add_postbuild_clear_output TARGET_NAME)
  foreach(OUTPUT_DIR IN LISTS ARGN)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E remove_directory ${OUTPUT_DIR}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
    )
  endforeach()
endfunction()



# ---- Libraries ----
function(add_custom_target target_name)
    set(options "")
    set(oneValueArgs TYPE LIBRARY_TYPE OUTPUT_DIR)
    set(multiValueArgs SOURCES INCLUDES LINK_LIBS DEFINES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT DEFINED ARG_TYPE)
        message(FATAL_ERROR "add_custom_target: Missing TYPE (LIBRARY or EXECUTABLE)")
    endif()

    if(NOT DEFINED ARG_SOURCES)
        message(FATAL_ERROR "add_custom_target: Missing SOURCES for ${target_name}")
    endif()

    if(ARG_TYPE STREQUAL "LIBRARY")
        if(DEFINED ARG_LIBRARY_TYPE)
            add_library(${target_name} ${ARG_LIBRARY_TYPE} ${ARG_SOURCES})
            message("    add_library(${target_name} ${ARG_LIBRARY_TYPE} ${ARG_SOURCES})")
        else()
            add_library(${target_name} ${ARG_SOURCES})  # default to STATIC/whatever global default is
            message("    add_library(${target_name} ${ARG_SOURCES})")
        endif()
        
    elseif(ARG_TYPE STREQUAL "EXECUTABLE")
        add_executable(${target_name} ${ARG_SOURCES})
        message("    add_executable(${target_name} ${ARG_SOURCES})")
    else()
        message(FATAL_ERROR "add_custom_target: TYPE must be LIBRARY or EXECUTABLE")
    endif()

    if(DEFINED ARG_INCLUDES)
        target_include_directories(${target_name} PUBLIC ${ARG_INCLUDES})
        message("    target_include_directories(${target_name} PUBLIC ${ARG_INCLUDES})")
    endif()

    if(DEFINED ARG_LINK_LIBS)
        target_link_libraries(${target_name} PUBLIC ${ARG_LINK_LIBS})
        message("    target_link_libraries(${target_name} PUBLIC ${ARG_LINK_LIBS})")
    endif()

    if(DEFINED ARG_DEFINES)
        target_compile_definitions(${target_name} PUBLIC ${ARG_DEFINES})
        message("    target_compile_definitions(${target_name} PUBLIC ${ARG_DEFINES})")
    endif()

    if(ARG_TYPE STREQUAL "EXECUTABLE" AND DEFINED ARG_OUTPUT_DIR)
        add_postbuild_clear_output(${target_name} ${ARG_OUTPUT_DIR})
        message("    add_postbuild_clear_output(${target_name} ${ARG_OUTPUT_DIR})")
    endif()
endfunction()



# ---- Targets ----
## Debug Defines
set(DEBUG_DEFINES
    ENABLE_DEBUG_IMAGES
    DEBUG_IMAGE_DIR="${TEST_OUTPUT_DIRECTORY}/debug"
    TEST_OUTPUT_DIRECTORY="${TEST_OUTPUT_DIRECTORY}"
    TEST_OUTPUT_ARCHIVE_DIRECTORY="${TEST_OUTPUT_ARCHIVE_DIRECTORY}"
)
message("==> DEBUG_DEFINES = \"${DEBUG_DEFINES}\"")


## C++ Code Files
set(CPU_SRC src/cpp/shape_detector_cpp.cpp src/shared/shared_utils.cpp src/cpp/utils_cpp.cpp)
set(CPU_ICLUDES ${CMAKE_SOURCE_DIR}/include)
set(CPU_LINK_LIBS ${OpenCV_LIBS})
message("==> CPU_SRC = \"${CPU_SRC}\"")
message("==> CPU_ICLUDES = \"${CPU_ICLUDES}\"")
message("==> CPU_LINK_LIBS = \"${CPU_LINK_LIBS}\"")
add_custom_target(triangle_detector_cpu
    TYPE LIBRARY
    SOURCES ${CPU_SRC}
    INCLUDES ${CPU_ICLUDES}
    LINK_LIBS ${CPU_LINK_LIBS}
)

## C++ DEMO Code Files
set(CPU_LOGIC_SRC src/shared/shape_detector_logic.cpp)
message("==> CPU_LOGIC_SRC = \"${CPU_LOGIC_SRC}\"")
add_custom_target(triangle_detector_cpu_logic
    TYPE LIBRARY
    SOURCES ${CPU_SRC} ${CPU_LOGIC_SRC}
    INCLUDES ${CPU_ICLUDES}
    LINK_LIBS ${CPU_LINK_LIBS}
)

# C++ TEST Code FILES
set(DEBUG_DEFINES_CPU ${DEBUG_DEFINES})
message("==> DEBUG_DEFINES_CPU = \"${DEBUG_DEFINES_CPU}\"")
set(CPU_DEBUG_SRC src/debug.cpp)
message("==> CPU_DEBUG_SRC = \"${CPU_DEBUG_SRC}\"")
add_custom_target(triangle_detector_cpp_for_test
    TYPE LIBRARY
    SOURCES ${CPU_SRC} ${CPU_DEBUG_SRC}
    INCLUDES ${CPU_ICLUDES}
    LINK_LIBS ${CPU_LINK_LIBS}
    DEFINES ${DEBUG_DEFINES_CPU}
)



## CUDA OpenCV Code Files
set(CUDA_OPENCV_SRC src/cuda_opencv/shape_detector_cuda_opencv.cu src/shared/shared_utils.cpp src/shared/shared_utils.cu src/cuda_opencv/utils_cuda_opencv.cu)
set(CUDA_OPENCV_INCLUDES ${CMAKE_SOURCE_DIR}/include)
set(CUDA_OPENCV_LINK_LIBS ${OpenCV_LIBS} opencv_cudaimgproc cuda)
message("==> CUDA_OPENCV_SRC = \"${CUDA_OPENCV_SRC}\"")
message("==> CUDA_OPENCV_INCLUDES = \"${CUDA_OPENCV_INCLUDES}\"")
message("==> CUDA_OPENCV_LINK_LIBS = \"${CUDA_OPENCV_LINK_LIBS}\"")
add_custom_target(triangle_detector_cuda_opencv
    TYPE LIBRARY
    SOURCES ${CUDA_OPENCV_SRC}
    INCLUDES ${CUDA_OPENCV_INCLUDES}
    LINK_LIBS ${CUDA_OPENCV_LINK_LIBS}
)

# CUDA OpenCV TEST Code FILES
set(DEBUG_DEFINES_CUDA_OPENCV ${DEBUG_DEFINES})
message("==> DEBUG_DEFINES_CUDA_OPENCV = \"${DEBUG_DEFINES_CUDA_OPENCV}\"")
set(CUDA_OPENCV_DEBUG_SRC src/debug.cpp src/debug.cu)
message("==> CUDA_OPENCV_DEBUG_SRC = \"${CUDA_OPENCV_DEBUG_SRC}\"")
add_custom_target(triangle_detector_cuda_opencv_for_test
    TYPE LIBRARY
    SOURCES ${CUDA_OPENCV_SRC} ${CUDA_OPENCV_DEBUG_SRC}
    INCLUDES ${CUDA_OPENCV_INCLUDES}
    LINK_LIBS ${CUDA_OPENCV_LINK_LIBS}
    DEFINES ${DEBUG_DEFINES_CUDA_OPENCV}
)



#---- Executables ----
## CPU Demo
set(CPU_DEMO_SRC demo/demo_cpp.cpp)
set(CPU_DEMO_LINK_LIBS triangle_detector_cpu_logic)
set(CPU_DEMO_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/demo/out")
message("==> CPU_DEMO_SRC = \"${CPU_DEMO_SRC}\"")
message("==> CPU_DEMO_LINK_LIBS = \"${CPU_DEMO_LINK_LIBS}\"")
message("==> CPU_DEMO_OUTPUT_DIR = \"${CPU_DEMO_OUTPUT_DIR}\"")
add_custom_target(triangle_detector_cpp_demo
  TYPE EXECUTABLE
  SOURCES ${CPU_DEMO_SRC}
  LINK_LIBS ${CPU_DEMO_LINK_LIBS}
  OUTPUT_DIR ${CPU_DEMO_OUTPUT_DIR}
)



# ---- Tests ----
function(add_gtest_exec exe_name)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SOURCES LIBS_TO_LINK OUTPUT_DIRS DEPENDENCIES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT DEFINED ARG_SOURCES)
        message(FATAL_ERROR "add_gtest_exec: Missing SOURCES argument!")
    endif()

    add_executable(${exe_name} ${ARG_SOURCES})
    message("    add_executable(${exe_name} ${ARG_SOURCES})")

    target_link_libraries(${exe_name} PRIVATE gtest gtest_main)
    message("    target_link_libraries(${exe_name} PRIVATE gtest gtest_main)")

    if(DEFINED ARG_LIBS_TO_LINK)
        target_link_libraries(${exe_name} PRIVATE ${ARG_LIBS_TO_LINK})
        message("    target_link_libraries(${exe_name} PRIVATE ${ARG_LIBS_TO_LINK})")
    endif()

    if(DEFINED ARG_DEPENDENCIES)
        add_dependencies(${exe_name} ${ARG_DEPENDENCIES})

        # Optional: inject dependencies list into test binary (only if needed)
        string(REPLACE ";" "," BINARIES_STR "${ARG_DEPENDENCIES}")
        target_compile_definitions(${exe_name} PRIVATE ALL_TEST_BINARIES="${BINARIES_STR}")
        message("    add_dependencies(${exe_name} ${ARG_DEPENDENCIES})")
        message("    string(REPLACE \";\" \",\" BINARIES_STR \"${ARG_DEPENDENCIES}\")")
        message("    target_compile_definitions(${exe_name} PRIVATE ALL_TEST_BINARIES=\"${BINARIES_STR}\")")
    endif()

    gtest_discover_tests(${exe_name})
    message("    gtest_discover_tests(${exe_name})")

    if(DEFINED ARG_OUTPUT_DIRS)
        foreach(output_dir IN LISTS ARG_OUTPUT_DIRS)
            add_postbuild_clear_output(${exe_name} ${output_dir})
            message("    add_postbuild_clear_output(${exe_name} ${output_dir})")
        endforeach()
    endif()
endfunction()


## CPU Tests
set(CPU_TEST_SRC tests/triangle_cpp_test.cpp)
set(CPU_TEST_DEPENDENCIES triangle_detector_cpp_for_test)
set(CPU_TEST_OUTPUT_DIRECTORIES ${TEST_OUTPUT_DIRECTORY} ${TEST_OUTPUT_ARCHIVE_DIRECTORY})
message("==> CPU_TEST_SRC = \"${CPU_TEST_SRC}\"")
message("==> CPU_TEST_DEPENDENCIES = \"${CPU_TEST_DEPENDENCIES}\"")
message("==> CPU_TEST_OUTPUT_DIRECTORIES = \"${CPU_TEST_OUTPUT_DIRECTORIES}\"")
add_gtest_exec(
  triangle_detector_cpp_tests
  SOURCES ${CPU_TEST_SRC}
  LIBS_TO_LINK ${CPU_TEST_DEPENDENCIES}
  OUTPUT_DIRS ${CPU_TEST_OUTPUT_DIRECTORIES}
)


## CUDA Tests
set(CUDA_OPENCV_TEST_SRC tests/triangle_cuda_opencv_test.cpp)
set(CUDA_OPENCV_TEST_DEPENDENCIES triangle_detector_cuda_opencv_for_test)
set(CUDA_OPENCV_TEST_OUTPUT_DIRECTORIES ${TEST_OUTPUT_DIRECTORY} ${TEST_OUTPUT_ARCHIVE_DIRECTORY})
message("==> CUDA_OPENCV_TEST_SRC = \"${CUDA_OPENCV_TEST_SRC}\"")
message("==> CUDA_OPENCV_TEST_DEPENDENCIES = \"${CUDA_OPENCV_TEST_DEPENDENCIES}\"")
message("==> CUDA_OPENCV_TEST_OUTPUT_DIRECTORIES = \"${CUDA_OPENCV_TEST_OUTPUT_DIRECTORIES}\"")
add_gtest_exec(
  triangle_detector_cuda_opencv_tests
  SOURCES ${CUDA_OPENCV_TEST_SRC}
  LIBS_TO_LINK ${CUDA_OPENCV_TEST_DEPENDENCIES}
  OUTPUT_DIRS ${CUDA_OPENCV_TEST_OUTPUT_DIRECTORIES}
)


set(ALL_TESTS_SRC tests/run_all_tests.cpp)
set(ALL_TESTS_DEPENDENCIES triangle_detector_cpp_tests triangle_detector_cuda_opencv_tests)
set(ALL_TESTS_OUTPUT_DIRECTORIES ${TEST_OUTPUT_DIRECTORY} ${TEST_OUTPUT_ARCHIVE_DIRECTORY}) #additional output directory list
message("==> ALL_TESTS_SRC = \"${ALL_TESTS_SRC}\"")
message("==> ALL_TESTS_DEPENDENCIES = \"${ALL_TESTS_DEPENDENCIES}\"")
message("==> ALL_TESTS_OUTPUT_DIRECTORIES = \"${ALL_TESTS_OUTPUT_DIRECTORIES}\"")
add_gtest_exec(
  run_all_tests
  SOURCES ${ALL_TESTS_SRC}
  DEPENDENCIES ${ALL_TESTS_DEPENDENCIES}
  OUTPUT_DIRS ${ALL_TESTS_OUTPUT_DIRECTORIES}
)

# also set the ALL_TESTS_DEPENDENCIES inside the tests/run_all_tests.cpp




# ---- Profiling ----



