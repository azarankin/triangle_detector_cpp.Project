#pragma once
#include <opencv2/opencv.hpp>
#include <string>

#ifdef ENABLE_DEBUG_IMAGES
  #define SAVE_DEBUG_IMAGE(image) save_debug_image(__func__, image)
  #define SAVE_DEBUG_IMAGE_AS(name, image) save_debug_image(name, image)
#else
  #define SAVE_DEBUG_IMAGE(image) ((void)0)
  #define SAVE_DEBUG_IMAGE_AS(name, image) ((void)0)
#endif

#ifndef DEBUG_IMAGE_DIR
#define DEBUG_IMAGE_DIR DEBUG_IMAGE_DEFAULT_DIR
#endif



void save_debug_image(const std::string& name, const cv::Mat& img);




// use SAVE_DEBUG_IMAGE(name, image);