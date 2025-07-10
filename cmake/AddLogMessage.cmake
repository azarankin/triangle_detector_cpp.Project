
# ---- Utility: Logging ----
string(TIMESTAMP NOW_TIMESTAMP "%Y-%m-%d %H:%M:%S")

function(file_writing_check file_path)
    if(NOT ${file_path} MATCHES "log")
            message(FATAL_ERROR "You request to rewrite a file that not contains 'log', ${LOG_FILE_PATH}")
    endif()

    if(NOT ${file_path} MATCHES "${CMAKE_SOURCE_DIR}")
            message(FATAL_ERROR "You request to remove a directory from path that not contains '${CMAKE_SOURCE_DIR}' ${ARGN}")
    endif()
endfunction()

function(log_message msg)
    file_writing_check(${LOG_FILE_PATH})
    file(APPEND ${LOG_FILE_PATH} "${msg}\n")
endfunction()

function(error_message msg)
    if(${file_path} MATCHES "log" OR ${file_path} MATCHES "${CMAKE_SOURCE_DIR}")
        message(FATAL_ERROR "${msg}")
    endif()
    file_writing_check(${ERROR_FILE_PATH})
    message(FATAL_ERROR "${msg}")
    file(APPEND ${ERROR_FILE_PATH} "${NOW_TIMESTAMP} -- ${msg}\n")
endfunction()

function(reset_log_file)
    file_writing_check(${LOG_FILE_PATH})
    file(WRITE ${LOG_FILE_PATH} "=== CMake Build Log ===\nCompiling: ${NOW_TIMESTAMP}\n=======================\n\n")
    message(STATUS "CMake Logs started at ${NOW_TIMESTAMP}")
    message(STATUS "Log file: ${LOG_FILE_PATH}")
endfunction()


function(reset_error_file)
    file_writing_check(${ERROR_FILE_PATH})
    file(WRITE ${ERROR_FILE_PATH} "=== CMake Errors ===\n\n")
    message(STATUS "CMake Logs started at ${NOW_TIMESTAMP}")
    message(STATUS "Error file: ${ERROR_FILE_PATH}")
endfunction()


