# ---- Project Output Directory Clearing ----
include(${CMAKE_SOURCE_DIR}/cmake/AddLogMessage.cmake) # Log the start of the configuration
function(add_postbuild_clear_output TARGET_NAME)
    log_message("add_postbuild_clear_output: ${TARGET_NAME}
        OUTPUT_DIRS: ${ARGN}
    ")
    foreach(OUTPUT_DIR IN LISTS ARGN)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${OUTPUT_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
    )
    endforeach()
endfunction()

