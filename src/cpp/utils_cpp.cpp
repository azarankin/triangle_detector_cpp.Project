#include <utils_cpp.h>
#include <debug.h>



std::string utils_print_message() 
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
    // cv::adaptiveThreshold(src, dst, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, 5);
    // SAVE_DEBUG_IMAGE(dst);
    // גירסת CPU שמחקה את האלגוריתם של CUDA
    // (1) ממוצע לוקאלי – boxFilter CPU
    cv::Mat mean_local;
    cv::boxFilter(src, mean_local, src.type(), cv::Size(11, 11));

    // (2) מחסרים C מהמטריצה
    cv::Mat threshMat;
    cv::subtract(mean_local, cv::Scalar::all(5), threshMat);

    // (3) משווים: src > threshMat ? 255 : 0 (THRESH_BINARY)
    cv::compare(src, threshMat, dst, cv::CMP_GT); // dst = mask
    // dst יוצא בינארי (0/255)
    
    SAVE_DEBUG_IMAGE(dst);
}


void threshold_filter(const cv::Mat& src, cv::Mat& dst)
{
    cv::threshold(src, dst, 127, 255, cv::THRESH_BINARY);
    SAVE_DEBUG_IMAGE(dst);
}
