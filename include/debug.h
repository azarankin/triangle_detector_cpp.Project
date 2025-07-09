#pragma once
#include <opencv2/opencv.hpp>


#ifdef ENABLE_DEBUG_IMAGES
#define SAVE_DEBUG_IMAGE(name, image) save_debug_image(name, img)
#else
#define SAVE_DEBUG_IMAGE(name, image) ((void)0)
#endif


void save_debug_image(const std::string& name, const cv::Mat& img);

