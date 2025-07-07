#pragma once
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>


#include <shared/utils.h>
#include <shared/utils.cuh>
#include <debug.cuh>

void cuda_adaptive_threshold(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int blockSize, double C);

void cuda_gray_filter(cv::cuda::GpuMat& gpu_frame ,cv::cuda::GpuMat& gpu_gray);

void cuda_gaussian_blur_filter(cv::cuda::GpuMat& gpu_gray, cv::cuda::GpuMat& gpu_blur);

void cuda_adaptive_threshold_filter(cv::cuda::GpuMat& gpu_blur, cv::cuda::GpuMat& gpu_thresh);

void cuda_threshold_filter(cv::cuda::GpuMat& gpu_blur, cv::cuda::GpuMat& gpu_thresh);

