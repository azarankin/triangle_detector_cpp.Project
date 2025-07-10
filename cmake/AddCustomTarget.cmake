
# ---- Project Target compilation ----
include(${CMAKE_SOURCE_DIR}/cmake/AddLogMessage.cmake) # Log the start of the configuration
function(add_custom_target target_name)
    set(options "")
    set(oneValueArgs TYPE LIBRARY_TYPE OUTPUT_DIR)
    set(multiValueArgs SOURCES INCLUDES LINK_LIBS DEFINES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    log_message("add_custom_target: ${target_name}
        TYPE: ${ARG_TYPE}
        SOURCES: ${ARG_SOURCES}
        LIBRARY_TYPE: ${ARG_LIBRARY_TYPE}
        INCLUDES: ${ARG_INCLUDES}
        LINK_LIBS: ${ARG_LINK_LIBS}
        DEFINES: ${ARG_DEFINES}
        OUTPUT_DIR: ${ARG_OUTPUT_DIR}
    ")

    if(NOT DEFINED ARG_TYPE)
        error_message("add_custom_target: Missing TYPE (LIBRARY or EXECUTABLE) for target ${target_name}")
        message(FATAL_ERROR "add_custom_target: Missing TYPE (LIBRARY or EXECUTABLE)")
        
    endif()

    if(NOT DEFINED ARG_SOURCES)
        error_message("add_custom_target: Missing SOURCES for target ${target_name}")
        message(FATAL_ERROR "add_custom_target: Missing SOURCES for ${target_name}")
    endif()

    if(ARG_TYPE STREQUAL "LIBRARY")
        if(DEFINED ARG_LIBRARY_TYPE)
            add_library(${target_name} ${ARG_LIBRARY_TYPE} ${ARG_SOURCES})
        else()
            add_library(${target_name} ${ARG_SOURCES})  # default to STATIC/whatever global default is
        endif()
    elseif(ARG_TYPE STREQUAL "EXECUTABLE")
        add_executable(${target_name} ${ARG_SOURCES})
    else()
        error_message("add_custom_target: TYPE must be LIBRARY or EXECUTABLE for target ${target_name}")
        message(FATAL_ERROR "add_custom_target: TYPE must be LIBRARY or EXECUTABLE")
    endif()

    if(DEFINED ARG_INCLUDES)
        target_include_directories(${target_name} PUBLIC ${ARG_INCLUDES})
    endif()

    if(DEFINED ARG_LINK_LIBS)
        target_link_libraries(${target_name} PUBLIC ${ARG_LINK_LIBS})
    endif()

    if(DEFINED ARG_DEFINES)
        target_compile_definitions(${target_name} PUBLIC ${ARG_DEFINES})
    endif()

    if(ARG_TYPE STREQUAL "EXECUTABLE" AND DEFINED ARG_OUTPUT_DIR)
        add_postbuild_clear_output(${target_name} ${ARG_OUTPUT_DIR})
    endif()
endfunction()

