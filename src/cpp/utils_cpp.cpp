#include <utils_cpp.h>
#include <debug.h>



std::string utils_print_message_cpp_utils() 
{
    std::cout << "\t\t\t\t\t\t\tPrinting from C++ Utils library!" << std::endl;
    return "cpp_utils";
}



void drawEquilateralTriangle(cv::Mat& img, cv::Size size, int margin)
{
    cv::Point pt1(size.width / 2, margin);                 // top
    cv::Point pt2(margin, size.height - margin);           // left
    cv::Point pt3(size.width - margin, size.height - margin); // right

    std::vector<cv::Point> triangle_cnt = {pt1, pt2, pt3};

    cv::drawContours(img, std::vector<std::vector<cv::Point>>{triangle_cnt}, 0, cv::Scalar(0, 0, 0), cv::FILLED);
}



void gray_filter(const cv::Mat& src, cv::Mat& dst)
{
    cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
    SAVE_DEBUG_IMAGE(dst);
}


void gaussian_blur_filter(const cv::Mat& src, cv::Mat& dst)
{
    // השתמש בחישוב אוטומטי של sigma כמו הגירסה המקורית
    cv::GaussianBlur(src, dst, cv::Size(5, 5), 0, 0, cv::BORDER_REFLECT101);
    SAVE_DEBUG_IMAGE(dst);
}


void adaptive_threshold_filter(const cv::Mat& src, cv::Mat& dst)
{
    cv::adaptiveThreshold(src, dst, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, 5);
    SAVE_DEBUG_IMAGE(dst);
}


void threshold_filter(const cv::Mat& src, cv::Mat& dst)
{
    cv::threshold(src, dst, 127, 255, cv::THRESH_BINARY);
    SAVE_DEBUG_IMAGE(dst);
}
