#include <gtest/gtest.h>

#include "test_logic.h" // #include <gtest/gtest.h> #include <opencv2/opencv.hpp> #include <filesystem> #include <string>

// the flag ENABLE_DEBUG_SAVE is defined (in CMakeLists.txt file)
#include "../shape_detector_cuda_opencv.cuh"

#include <opencv2/opencv.hpp>


/* OpenCV with CUDA application*/

class CUDAOpenCVTest : public TriangleImageTest
{
    void SetUp() override
    {
        #ifdef ENABLE_DEBUG_SAVE
            g_debug_config.enable = true;
        #else
            g_debug_config.enable = false;
        #endif

        #ifdef DEBUG_OUTPUT_DIR
            g_debug_config.output_dir = DEBUG_OUTPUT_DIR;
        #else
            g_debug_config.output_dir = current_output;
        #endif
    }
    void TearDown() override {
        g_debug_config.prefix.erase();
        // פעולות ניקוי או שמירה אחרי הטסט
    }

};


TEST_F(CUDAOpenCVTest, CudaOpenCVMessage) 
{
    
    ASSERT_EQ(print_message(), "cuda_opencv"); 
}




TEST_F(CUDAOpenCVTest, GenerateTriangle) /*cuda finish*/
{
    create_triangle_image(new_triangle_img);
    cv::Mat expected = cv::imread(this->expected_triangle_img, cv::IMREAD_COLOR);
    cv::Mat result = cv::imread(new_triangle_img, cv::IMREAD_COLOR);
    //img_save_the_difference_between_images(current_output / "diff.png", expected, result);
    image_similarity_asserts(expected, result);
}





TEST_F(CUDAOpenCVTest, DetectTriangleContour) 
{
    g_debug_config.prefix = "cuda_detect_contour";
    auto template_contour = find_shape_contour(triangle_img);
    cv::Mat triangle_image = cv::imread(triangle_img, cv::IMREAD_COLOR);
    auto result = draw_contour_on_image(triangle_image, template_contour);
    cv::Mat expected = cv::imread(expected_triangle_contours_img, cv::IMREAD_COLOR);
    image_similarity_asserts(expected, result);
}

TEST_F(CUDAOpenCVTest, DetectTriangleContourOnAnother) 
{
    g_debug_config.prefix = "cuda_contour_compare";
    auto template_contour = find_shape_contour(triangle_img);
    cv::Mat other_triangle_image = cv::imread(other_triangle_img, cv::IMREAD_COLOR);
    auto detected_contours = contour_compare(other_triangle_image, template_contour);
    auto result = draw_contour_on_image(other_triangle_image, detected_contours);
    cv::Mat expected = cv::imread(expected_other_triangle_contours_img, cv::IMREAD_COLOR);

    image_similarity_asserts(expected, result);
}
