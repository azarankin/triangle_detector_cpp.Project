#include <gtest/gtest.h>
#include <test_logic.h>  // #include <gtest/gtest.h> #include <opencv2/opencv.hpp> #include <filesystem> #include <string>
// the test functions
#include <utils_cpp.h>

/* Cpp application*/
class CppUtils : public UtilsTest
{
    void SetUp() override
    {
        fs::create_directories(current_output);
        input_image = get_gray_filtered_img(input_gray_img);
    }
    void TearDown() override 
    {
        std::string test_name = get_current_test_name() ;
        archive_directory(current_output, archive_current_output);
    }
protected:
    cv::Mat input_image;
    cv::Mat output;

    void image_similarity_asserts(const std::string& expected_path, const cv::Mat& result, double tol = 1e-6)
    {
        // Validate the expected image path first
        ASSERT_FALSE(expected_path.empty()) << "Expected gray filter image path is empty.";
        ASSERT_TRUE(fs::exists(expected_path)) << "Expected gray filter image does not exist: " + expected_path;
        
        cv::Mat expected_image = get_gray_filtered_img(expected_path);
        //ASSERT_FALSE(expected_image.empty()) << "Failed to load expected image: " << expected_path;
        //ASSERT_FALSE(result.empty()) << "Result image is empty.";
        TestLogic::image_similarity_asserts(expected_image, result, tol);
    }

    cv::Mat get_gray_filtered_img(const std::string& input_gray_img) 
    {
        return cv::imread(input_gray_img, cv::IMREAD_GRAYSCALE);
    }
};


/*clean C++ Utils code Tests*/
//tested on gray image

TEST_F(CppUtils, CppCompiledUtilsMessage) 
{
    ASSERT_EQ(utils_print_message(), "cpp_utils"); 
}


TEST_F(CppUtils, GrayFilter) 
{ 
    cv::Mat input_image_color = cv::imread(original_img, cv::IMREAD_COLOR);
    gray_filter(input_image_color, output);
    image_similarity_asserts(expected_gray_filter_img, output);
}


TEST_F(CppUtils, GaussianBlueFilter) 
{ 
    gaussian_blur_filter(input_image, output);
    image_similarity_asserts(expected_gaussian_blur_filter_img, output);
}


TEST_F(CppUtils, AdaptiveThresholdFilter) 
{ 
    adaptive_threshold_filter(input_image, output);
    image_similarity_asserts(expected_adaptive_threshold_filter_img, output);
}


TEST_F(CppUtils, ThresholdFilter) 
{ 
    threshold_filter(input_image, output);
    image_similarity_asserts(expected_threshold_filter_img, output);
}


/* drawEquilateralTriangle tested with triangle tests */