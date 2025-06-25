
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include "../shape_detector_cpp.h"


namespace fs = std::filesystem;
fs::path source_file = __FILE__;
fs::path dir = source_file.parent_path();


fs::path REFERENCE_INPUT = dir / "reference_input";
const std::string triangle_img = "triangle.png";
const std::string other_triangle_img = "other_triangle.png";
fs::path triangle_img_path = REFERENCE_INPUT / triangle_img;
fs::path other_triangle_img_path = REFERENCE_INPUT / other_triangle_img;

fs::path REFERENCE_OUTPUT = dir / "reference_output";
const std::string triangle_contours_img = "triangle_contours.png";
const std::string other_triangle_contours_img = "other_triangle_contours.png";
cv::Mat expected_triangle_img = cv::imread(fs::path(REFERENCE_OUTPUT / triangle_img), cv::IMREAD_COLOR);
cv::Mat expected_triangle_contours_img = cv::imread(fs::path(REFERENCE_OUTPUT / triangle_contours_img), cv::IMREAD_COLOR);
cv::Mat expected_other_triangle_contours_img = cv::imread(fs::path(REFERENCE_OUTPUT / other_triangle_contours_img), cv::IMREAD_COLOR);

fs::path CURRENT_OUTPUT = dir / "current_output";
fs::path new_triangle_img_path = CURRENT_OUTPUT / triangle_img;



TEST(SimpleTest, BasicCheck) {
    ASSERT_EQ(1 + 1, 2);
}
// Check if file do exist

void image_similarity_asserts(const cv::Mat& expected, const cv::Mat& result)
{
    ASSERT_FALSE(expected.empty()) << "Reference image is missing!";

    // First check if the sizes and the type is the same
    ASSERT_EQ(result.size(), expected.size()) << "Result image output is not same size";
    ASSERT_EQ(result.type(), expected.type()) << "Result image output is not same type";

    // Calc the differences (Norm)
    double diff = cv::norm(result, expected, cv::NORM_L2);
    // 0 < 1  e-6 is good enogh range
    ASSERT_LT(diff, 1e-6) << "Result image output is not similar enough!";
}


TEST(TriangleImageTest, GenerateSampleTriangle) 
{
    
    create_triangle_image(new_triangle_img_path); 
    cv::Mat result = cv::imread(new_triangle_img_path, cv::IMREAD_COLOR);

    image_similarity_asserts(expected_triangle_img, result);
}

TEST(TriangleImageTest, CheckTriangleCountour) 
{
    auto template_contour = find_shape_contour(triangle_img_path);

    cv::Mat triangle_image = cv::imread(triangle_img_path, cv::IMREAD_COLOR);
    auto result = draw_contour_on_image(triangle_image, template_contour);

    //cv::imwrite(fs::path(CURRENT_OUTPUT / triangle_img), result);
    image_similarity_asserts(expected_triangle_contours_img, result);
}





TEST(TriangleImageTest, ApplyTriangleCountoursOnAnother) 
{
    auto template_contour = find_shape_contour(triangle_img_path);

    cv::Mat triangle_image = cv::imread(triangle_img_path, cv::IMREAD_COLOR);

    cv::Mat other_triangle_image = cv::imread(other_triangle_img_path, cv::IMREAD_COLOR);
    //imshow("", other_triangle_frame_img);
    auto detected_contours = contour_compare(other_triangle_image, template_contour);
    auto result = draw_contour_on_image(other_triangle_image, detected_contours);
    
    image_similarity_asserts(expected_other_triangle_contours_img, result);
}


