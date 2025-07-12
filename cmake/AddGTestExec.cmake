# ---- GTest Executable ----
include(${CMAKE_SOURCE_DIR}/cmake/AddLogMessage.cmake) # Log the start of the configuration
include(${CMAKE_SOURCE_DIR}/cmake/AddPostBuildClearOutput.cmake) # Include post-build clear output function, add_postbuild_clear_output
function(add_gtest_exec exe_name)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SOURCES LIBS_TO_LINK OUTPUT_DIRS DEPENDENCIES INCLUDES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    log_message("add_gtest_exec: ${exe_name}
        SOURCES: ${ARG_SOURCES}
        LIBS_TO_LINK: ${ARG_LIBS_TO_LINK}
        OUTPUT_DIRS: ${ARG_OUTPUT_DIRS}
        DEPENDENCIES: ${ARG_DEPENDENCIES}
        INCLUDES: ${ARG_INCLUDES}
    ")

    if(NOT DEFINED ARG_SOURCES)
        message(FATAL_ERROR "add_gtest_exec: Missing SOURCES argument!")
        error_message("add_gtest_exec: Missing SOURCES argument for executable ${exe_name}")
    endif()

    add_executable(${exe_name} ${ARG_SOURCES})

    target_link_libraries(${exe_name} PRIVATE gtest gtest_main)

    if(DEFINED ARG_LIBS_TO_LINK)
        target_link_libraries(${exe_name} PRIVATE ${ARG_LIBS_TO_LINK})
    endif()

    if(DEFINED ARG_DEPENDENCIES)
        add_dependencies(${exe_name} ${ARG_DEPENDENCIES})
        # Optional: inject dependencies list into test binary (only if needed)
        string(REPLACE ";" "," BINARIES_STR "${ARG_DEPENDENCIES}")
        target_compile_definitions(${exe_name} PRIVATE ALL_TEST_BINARIES="${BINARIES_STR}")
    endif()


    gtest_discover_tests(${exe_name})

    if(DEFINED ARG_OUTPUT_DIRS)
        foreach(output_dir IN LISTS ARG_OUTPUT_DIRS)
            add_postbuild_clear_output(${exe_name} ${output_dir})
        endforeach()
    endif()

    if(DEFINED ARG_INCLUDES)
        target_include_directories(${exe_name} PRIVATE ${ARG_INCLUDES})
    endif()

endfunction()
