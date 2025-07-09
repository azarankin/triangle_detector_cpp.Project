
#pragma once
#include <shared_utils.h>
#include <debug.h>

void drawEquilateralTriangle(cv::Mat& img, cv::Size size, int margin);

void gray_filter(cv::Mat& target_frame ,cv::Mat& target_gray);

void gaussian_blur_filter(cv::Mat& target_gray, cv::Mat& blured);

void adaptive_threshold_filter(cv::Mat& blured, cv::Mat& thresh2);

void threshold_filter(cv::Mat& blured, cv::Mat& thresh1);

