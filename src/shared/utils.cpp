#include <shared/utils.h>

std::vector<cv::Point> get_second_largest_contour(std::vector<std::vector<cv::Point>>& contours_template)
{
    std::sort(contours_template.begin(), contours_template.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return cv::contourArea(a) > cv::contourArea(b);
        });

    if (contours_template.size() < 2)
        throw std::runtime_error("Template image must contain at least two contours.");

    return contours_template[1]; // Return the second largest contour
}


void find_contours(std::vector<std::vector<cv::Point>>& contours_target, cv::Mat& thresh2)
{
    cv::findContours(thresh2, contours_target, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
}


std::vector<std::vector<cv::Point>> find_closest_contours(cv::Mat& target_frame, const std::vector<cv::Point>& template_contour, std::vector<std::vector<cv::Point>>& contours_target, double match_threshold, double min_area_ratio, double max_area_ratio)
{
        // הגדרות סף שטח
    double frame_area = static_cast<double>(target_frame.rows * target_frame.cols);
    double min_area = frame_area * min_area_ratio;
    double max_area = frame_area * max_area_ratio;

    std::vector<std::vector<cv::Point>> closest_contours;

    // חיפוש קונטורים דומים
    for (const auto& c : contours_target) {
        double match = cv::matchShapes(template_contour, c, cv::CONTOURS_MATCH_I3, 0.0);
        if (match < match_threshold) {
            double area = cv::contourArea(c);
            if (area < min_area || area > max_area)
                continue; // קטן/גדול מדי = רעש
            closest_contours.push_back(c);
        }
    }

    return closest_contours;
}



cv::Mat draw_contour_on_image(cv::Mat& img, const std::vector<cv::Point>& contour, const cv::Scalar& color, int thickness)
{
    if (img.empty())
        throw std::runtime_error("Failed to load image");

    std::vector<std::vector<cv::Point>> contours{contour};
    cv::drawContours(img, contours, 0, color, thickness);
    return img;
}


cv::Mat draw_contour_on_image(cv::Mat& img, const std::vector<std::vector<cv::Point>>& contours, const cv::Scalar& color, int thickness) 
{
    if (img.empty())
        throw std::runtime_error("Failed to load image");
    cv::drawContours(img, contours, -1, color, thickness);
    return img;
}


