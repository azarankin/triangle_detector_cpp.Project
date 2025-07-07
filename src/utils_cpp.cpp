#include <utils_cpp.h>


void drawEquilateralTriangle(cv::Mat& img, cv::Size size, int margin)
{
    cv::Point pt1(size.width / 2, margin);                 // top
    cv::Point pt2(margin, size.height - margin);           // left
    cv::Point pt3(size.width - margin, size.height - margin); // right

    std::vector<cv::Point> triangle_cnt = {pt1, pt2, pt3};

    cv::drawContours(img, std::vector<std::vector<cv::Point>>{triangle_cnt}, 0, cv::Scalar(0, 0, 0), cv::FILLED);
}



void gray_filter(cv::Mat& target_frame ,cv::Mat& target_gray)
{
    cv::cvtColor(target_frame, target_gray, cv::COLOR_BGR2GRAY);
}


void gaussian_blur_filter(cv::Mat& target_gray, cv::Mat& blured)
{
    cv::GaussianBlur(target_gray, blured, cv::Size(5, 5), 0);
}


void adaptive_threshold_filter(cv::Mat& blured, cv::Mat& thresh2)
{
    cv::adaptiveThreshold(blured, thresh2, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, 5);
}



void threshold_filter(cv::Mat& blured, cv::Mat& thresh1)
{
    cv::threshold(blured, thresh1, 127, 255, 0);
}


