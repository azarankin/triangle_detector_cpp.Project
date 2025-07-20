#pragma once
#include <opencv2/opencv.hpp>
#include <shape_detector_common.h>


void imshow(const std::string& title, const cv::Mat& img);

void run_camera(
    const std::vector<cv::Point>& template_contour,
    int camid = 0,
    int display_width = 800,
    int display_height = 600
);
