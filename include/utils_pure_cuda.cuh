#pragma once
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>



#include <shared_utils.cuh>
#include <debug.cuh>


std::string utils_print_message();


void cuda_gray_filter(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);


void cuda_gaussian_blur_filter(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);


void cuda_adaptive_threshold_filter(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);


void cuda_threshold_filter(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);

