# /benchmarks/CMakeLists.txt

add_library(benchmark_utils_lib STATIC utils.cu)
target_link_libraries(benchmark_utils_lib PUBLIC cpp_utils_lib cuda_opencv_utils_lib pure_cuda_utils_lib)
#target_link_libraries(benchmark_utils_lib PUBLIC ${OpenCV_LIBS} opencv_cudaimgproc cuda)
target_include_directories(benchmark_utils_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(utils_benchmark benchmark.cu)
target_link_libraries(utils_benchmark PRIVATE benchmark_utils_lib)

add_postbuild_clear_output(utils_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/out)