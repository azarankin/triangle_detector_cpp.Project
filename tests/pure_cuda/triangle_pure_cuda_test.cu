#include <gtest/gtest.h>
#include <test_logic.h> // #include <opencv2/opencv.hpp> #include <filesystem> #include <string>

// the flag ENABLE_DEBUG_SAVE is defined (in CMakeLists.txt file)
#include <shape_detector_common.h>


/* CUDA OpenCV Triangle Detector application*/

class PureCUDATest : public TriangleImageTest
{
    void SetUp() override
    {
        
        fs::create_directories(current_output);
    }
    void TearDown() override 
    {
        std::string test_name = get_current_test_name() ;
        archive_directory(current_output, archive_current_output);
        
    }

};


TEST_F(PureCUDATest, CudaOpenCVMessage) 
{
    
    ASSERT_EQ(triangle_print_message(), "pure_cuda_triangle_detector"); 
}




TEST_F(PureCUDATest, GenerateTriangle) /*cuda finish*/
{
    create_triangle_image(new_triangle_img);
    cv::Mat expected = cv::imread(this->expected_triangle_img, cv::IMREAD_COLOR);
    cv::Mat result = cv::imread(new_triangle_img, cv::IMREAD_COLOR);
    //img_save_the_difference_between_images(current_output / "diff.png", expected, result);
    image_visual_similarity_asserts(expected, result);
}





TEST_F(PureCUDATest, DetectTriangleContour) 
{
    auto template_contour = find_shape_contour(triangle_img);
    cv::Mat triangle_image = cv::imread(triangle_img, cv::IMREAD_COLOR);
    auto result = draw_contour_on_image(triangle_image, template_contour);
    cv::Mat expected = cv::imread(expected_triangle_contours_img, cv::IMREAD_COLOR);
    image_visual_similarity_asserts(expected, result);
}

TEST_F(PureCUDATest, DetectTriangleContourOnAnother) 
{
    auto template_contour = find_shape_contour(triangle_img);
    cv::Mat other_triangle_image = cv::imread(other_triangle_img, cv::IMREAD_COLOR);
    auto detected_contours = contour_compare(other_triangle_image, template_contour);
    auto result = draw_contour_on_image(other_triangle_image, detected_contours);
    cv::Mat expected = cv::imread(expected_other_triangle_contours_img, cv::IMREAD_COLOR);

    image_visual_similarity_asserts(expected, result);
}


