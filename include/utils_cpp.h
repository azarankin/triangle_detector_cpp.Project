#pragma once
#include <shared_utils.h>
#include <debug.h>


std::string utils_print_message();


void gray_filter(const cv::Mat& src, cv::Mat& dst);


void gaussian_blur_filter(const cv::Mat& src, cv::Mat& dst);


void adaptive_threshold_filter(const cv::Mat& src, cv::Mat& dst);


void threshold_filter(const cv::Mat& src, cv::Mat& dst);


void drawEquilateralTriangle(cv::Mat& img, cv::Size size, int margin);
