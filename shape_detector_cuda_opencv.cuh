#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <filesystem>
namespace fs = std::filesystem;



std::string print_message();

void imshow(const std::string& title, const cv::Mat& img);

std::vector<cv::Point> find_shape_contour(const std::string& address);

cv::Mat draw_contour_on_image(cv::Mat& img, const std::vector<cv::Point>& contour, const cv::Scalar& color = cv::Scalar(0,255,0), int thickness=3);

cv::Mat draw_contour_on_image(cv::Mat& img, const std::vector<std::vector<cv::Point>>& contours, const cv::Scalar& color = cv::Scalar(0, 255, 0), int thickness=3);


std::vector<std::vector<cv::Point>> contour_compare(
    cv::Mat& target_frame, 
    const std::vector<cv::Point>& template_contour,
    double match_threshold = 0.09, 
    double min_area_ratio = 0.0005, 
    double max_area_ratio = 0.80);
    
void run_camera(const std::vector<cv::Point>& template_contour, int camid = 0);

void create_triangle_image(const std::string& filename = "triangle.png",
                                 cv::Size size = {600, 600}, int margin = 150);


/* Debugging */

void save_debug_image(const std::string& name, const cv::Mat& img);

void save_debug_image(const std::string& name, const cv::cuda::GpuMat& gpu_img);

struct DebugConfig {
    bool enable = false;
    std::string output_dir;
    std::string prefix;
};

extern DebugConfig g_debug_config;
