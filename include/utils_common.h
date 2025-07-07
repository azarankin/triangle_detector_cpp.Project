#pragma once
#include <opencv2/opencv.hpp>
#include <vector>


cv::Mat draw_contour_on_image(cv::Mat& img, const std::vector<cv::Point>& contour, const cv::Scalar& color, int thickness);

cv::Mat draw_contour_on_image(cv::Mat& img, const std::vector<std::vector<cv::Point>>& contours, const cv::Scalar& color, int thickness) ;

std::vector<cv::Point> get_second_largest_contour(std::vector<std::vector<cv::Point>>& contours_template);

void find_contours(std::vector<std::vector<cv::Point>>& contours_target, cv::Mat& thresh2);

std::vector<std::vector<cv::Point>> find_closest_contours(cv::Mat& target_frame, const std::vector<cv::Point>& template_contour, std::vector<std::vector<cv::Point>>& contours_target, double match_threshold, double min_area_ratio, double max_area_ratio);

