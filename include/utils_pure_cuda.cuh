#pragma once
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>


#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <shared_utils.cuh>
#include <debug.cuh>


std::string utils_print_message_pure_cuda();


__global__ void bgr2gray_kernel(const uchar3* src, unsigned char* dst, int width, int height);

void cuda_pure_gray_filter_async(const unsigned char* h_src_bgr, unsigned char* h_dst_gray, int width, int height, cudaStream_t stream = 0);





void upload_gaussian_kernel_5x5();

__global__ void gaussian_blur_kernel(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int width, int height, int src_stride, int dst_stride);

void pure_cuda_gaussian_blur_filter( const unsigned char* h_src, unsigned char* h_dst, int width, int height, int src_stride, int dst_stride, cudaStream_t stream = 0);


__global__ void adaptive_threshold_kernel_11_5(
    const unsigned char* __restrict__ src, unsigned char* __restrict__ dst,
    int width, int height,
    int src_stride, int dst_stride);

void pure_cuda_adaptive_threshold(
    const unsigned char* h_src,
    unsigned char* h_dst,
    int width, int height,
    int src_stride, int dst_stride,
    cudaStream_t stream = 0);



__global__ void threshold_kernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int width, int height,
    int src_stride, int dst_stride,
    unsigned char thresh, unsigned char maxval
);

void pure_cuda_threshold_filter(
    const unsigned char* h_src,
    unsigned char* h_dst,
    int width, int height,
    int src_stride, int dst_stride,
    cudaStream_t stream = 0,
    unsigned char thresh = 127,
    unsigned char maxval = 255
);