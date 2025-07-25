#include <shape_detector_common.h>
#include <utils_cpp.h>


std::string triangle_print_message() 
{
    std::cout << "\t\t\t\t\t\t\tPrinting from C++ Triangle Detector library!" << std::endl;
    return "cpp_triangle_detector";
}


std::vector<cv::Point> find_shape_contour(const std::string& address)
{
    // טען תמונה
    cv::Mat templat_img = cv::imread(address);
    if (templat_img.empty())
        throw std::runtime_error("Failed to load image: " + address);

    // המרה לאפור
    cv::Mat templat_img_gray;
    gray_filter(templat_img ,templat_img_gray);

    // Gaussian blur
    cv::Mat blured;
    gaussian_blur_filter(templat_img_gray, blured);

    // Threshold
    cv::Mat thresh1;
    threshold_filter(blured, thresh1);

    // Find contours (CPU only)
    std::vector<std::vector<cv::Point>> contours_template;
    find_contours(contours_template, thresh1);

    // Sort contours by area
    auto second_largest_contour = get_second_largest_contour(contours_template);

    // Draw the second largest contour
    return second_largest_contour;
}





std::vector<std::vector<cv::Point>> contour_compare(
    cv::Mat& target_frame, 
    const std::vector<cv::Point>& template_contour,
    double match_threshold, 
    double min_area_ratio, 
    double max_area_ratio)
{
    // Gray filter
    cv::Mat target_gray;
    gray_filter(target_frame ,target_gray);

    // Gaussian blur
    cv::Mat blured;
    gaussian_blur_filter(target_gray, blured);

    // Adaptive threshold
    cv::Mat thresh2;
    adaptive_threshold_filter(blured, thresh2);

    // Contours (CPU)
    std::vector<std::vector<cv::Point>> contours_target;
    find_contours(contours_target, thresh2);

    std::vector<std::vector<cv::Point>> closest_contours = find_closest_contours(target_frame, template_contour, contours_target, match_threshold, min_area_ratio, max_area_ratio);

    return closest_contours;
}




void create_triangle_image(const std::string& filename, cv::Size size, int margin) 
{
    cv::Mat img(size, CV_8UC3, cv::Scalar(255, 255, 255));
    drawEquilateralTriangle(img, size, margin);
    cv::imwrite(filename, img);
    std::cout << "Triangle image saved as " << filename << std::endl;
}

