
# ---- Utility: Logging ----
string(TIMESTAMP NOW_TIMESTAMP "%Y-%m-%d %H:%M:%S")

function(log_message msg)
    file(APPEND ${LOG_FILE_PATH} "${msg}\n")
endfunction()

function(error_message msg)
    file(APPEND ${ERROR_FILE_PATH} "${NOW_TIMESTAMP} -- ${msg}\n")
endfunction()

function(reset_log_file)
    file(WRITE ${LOG_FILE_PATH} "=== CMake Build Log ===\nCompiling: ${NOW_TIMESTAMP}\n=======================\n\n")
    message(STATUS "CMake Logs started at ${NOW_TIMESTAMP}")
    message(STATUS "Log file: ${LOG_FILE_PATH}")
endfunction()


function(reset_error_file)
    file(WRITE ${ERROR_FILE_PATH} "=== CMake Errors ===\n\n")
    message(STATUS "CMake Logs started at ${NOW_TIMESTAMP}")
    message(STATUS "Error file: ${ERROR_FILE_PATH}")
endfunction()


