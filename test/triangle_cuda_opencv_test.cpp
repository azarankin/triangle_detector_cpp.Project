#include <gtest/gtest.h>

#include "test_logic.h" // #include <gtest/gtest.h> #include <opencv2/opencv.hpp> #include <filesystem> #include <string>
// the functions
#include "../shape_detector_cuda_opencv.cuh"




/* OpenCV with CUDA application*/

class CUDAOpenCVTest : public TriangleImageTest{};

TEST_F(CUDAOpenCVTest, CudaOpenCVMessage) 
{
    ASSERT_EQ(print_message(), "cuda_opencv"); 
}


TEST_F(CUDAOpenCVTest, GenerateTriangle) 
{
    create_triangle_image(new_triangle_img);
    cv::Mat expected = cv::imread(this->expected_triangle_img, cv::IMREAD_COLOR);
    cv::Mat result = cv::imread(new_triangle_img, cv::IMREAD_COLOR);
    image_similarity_asserts(expected, result);
}

TEST_F(CUDAOpenCVTest, DetectTriangleContour) 
{
    auto template_contour = find_shape_contour(triangle_img);
    cv::Mat triangle_image = cv::imread(triangle_img, cv::IMREAD_COLOR);
    auto result = draw_contour_on_image(triangle_image, template_contour);
    cv::Mat expected = cv::imread(expected_triangle_contours_img, cv::IMREAD_COLOR);
    image_similarity_asserts(expected, result);
}

TEST_F(CUDAOpenCVTest, DetectTriangleContourOnAnother) 
{
    auto template_contour = find_shape_contour(triangle_img);
    cv::Mat other_triangle_image = cv::imread(other_triangle_img, cv::IMREAD_COLOR);
    auto detected_contours = contour_compare(other_triangle_image, template_contour);
    auto result = draw_contour_on_image(other_triangle_image, detected_contours);
    cv::Mat expected = cv::imread(expected_other_triangle_contours_img, cv::IMREAD_COLOR);
    image_similarity_asserts(expected, result);
}
