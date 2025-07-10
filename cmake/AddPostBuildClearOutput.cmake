# ---- Project Output Directory Clearing ----
include(${CMAKE_SOURCE_DIR}/cmake/AddLogMessage.cmake) # Log the start of the configuration
function(add_postbuild_clear_output TARGET_NAME)
    #import checking
    if(NOT ARGN MATCHES "out")
            
            error_message("⚠️ You request to remove a directory from path that not contains 'out' or 'output' ${ARGN}")
            message(FATAL_ERROR "You request to remove a directory from path that not contains 'out' or 'output' ${ARGN}")
    endif()

    if(NOT ARGN MATCHES "${CMAKE_SOURCE_DIR}")            
            error_message("⚠️ You request to remove a directory from path that not contains '${CMAKE_SOURCE_DIR}' ${ARGN}")
            message(FATAL_ERROR "You request to remove a directory from path that not contains '${CMAKE_SOURCE_DIR}' ${ARGN}")
    endif()

    log_message("Directories to clear for target ${TARGET_NAME}: ${ARGN}")

    foreach(OUTPUT_DIR IN LISTS ARGN)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${OUTPUT_DIR}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
        )
    endforeach()
endfunction()

