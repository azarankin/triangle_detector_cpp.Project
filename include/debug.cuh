#pragma once
#include <debug.h>


void save_debug_image(const std::string& name, const cv::cuda::GpuMat& gpu_img);



// use SAVE_DEBUG_IMAGE(name, image);