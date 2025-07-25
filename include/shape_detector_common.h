#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <shared_utils.h>

namespace fs = std::filesystem;

//#include <shape_detector_logic.h>


std::string triangle_print_message();


std::vector<cv::Point> find_shape_contour(const std::string& address);


cv::Mat draw_contour_on_image(cv::Mat& img, const std::vector<cv::Point>& contour, const cv::Scalar& color = cv::Scalar(0,255,0), int thickness=3);


cv::Mat draw_contour_on_image(cv::Mat& img, const std::vector<std::vector<cv::Point>>& contours, const cv::Scalar& color = cv::Scalar(0, 255, 0), int thickness=3);


std::vector<std::vector<cv::Point>> contour_compare(
    cv::Mat& target_frame, 
    const std::vector<cv::Point>& template_contour,
    double match_threshold = 0.09, 
    double min_area_ratio = 0.0005, 
    double max_area_ratio = 0.80);

void create_triangle_image(const std::string& filename = "triangle.png",
                                 cv::Size size = cv::Size{600, 600}, int margin = 150);
    


void imshow(const std::string& title, const cv::Mat& img);
