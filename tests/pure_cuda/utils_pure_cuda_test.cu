#include <gtest/gtest.h>
#include <test_logic.h>  // #include <gtest/gtest.h> #include <opencv2/opencv.hpp> #include <filesystem> #include <string>
// the test functions
#include <utils_pure_cuda.cuh>

/* Cpp application*/
class PureCudaUtils : public UtilsTest
{
    void SetUp() override
    {
        fs::create_directories(current_output);
        input_image = get_gray_filtered_img(input_gray_img);
        ASSERT_FALSE(input_image.empty()) << "Failed to load input image: " << input_gray_img;
        //input_image_gpu.upload(input_image);
        //output_gpu = cv::cuda::GpuMat(input_image_gpu.size(), input_image_gpu.type());
        //output = cv::Mat(input_image_gpu.size(), input_image_gpu.type());
    }
    void TearDown() override 
    {
        std::string test_name = get_current_test_name() ;
        archive_directory(current_output, archive_current_output);
    }
protected:
    cv::Mat input_image; 
    cv::Mat output;

    void image_visual_similarity_asserts_for_gpu_output(const std::string& expected_path)
    {

        // Validate the expected image path first
        ASSERT_FALSE(expected_path.empty()) << "Expected gray filter image path is empty.";
        ASSERT_TRUE(fs::exists(expected_path)) << "Expected gray filter image does not exist: " + expected_path;
        
        cv::Mat expected_image = get_gray_filtered_img(expected_path);
        //ASSERT_FALSE(expected_image.empty()) << "Failed to load expected image: " << expected_path;
        //ASSERT_FALSE(result.empty()) << "Result image is empty.";
        TestLogic::image_visual_similarity_asserts(expected_image, output);
    }



    cv::Mat get_gray_filtered_img(const std::string& input_gray_img) 
    {
        return cv::imread(input_gray_img, cv::IMREAD_GRAYSCALE);
    }
};


/*clean C++ Utils code Tests*/
//tested on gray image

TEST_F(PureCudaUtils, CppCompiledUtilsMessage) 
{
    ASSERT_EQ(utils_print_message_pure_cuda(), "pure_cuda_utils"); 
}


TEST_F(PureCudaUtils, GrayFilter) 
{ 
    cv::Mat input_image_color = cv::imread(original_img, cv::IMREAD_COLOR);
    ASSERT_EQ(input_image_color.type(), CV_8UC3); // BGR input
    output.create(input_image_color.rows, input_image_color.cols, CV_8UC1);

    cuda_pure_gray_filter(input_image_color, output);

    image_save(debug_output / "pure_cuda_gray_filter.png", output);

    image_visual_similarity_asserts_for_gpu_output(expected_gray_filter_img);
}


TEST_F(PureCudaUtils, GaussianBlueFilter) 
{ 
    ASSERT_EQ(input_image.type(), CV_8UC1); // BGR input
    output.create(input_image.rows, input_image.cols, CV_8UC1);

    pure_cuda_gaussian_blur_filter(input_image, output);

    //image_save(debug_output / "pure_cuda_gaussian_blur_filter99.png", output);
    //output_gpu.download(output);
    image_save(debug_output / "pure_cuda_output_gaussian_blur_filter.png", output);

    // השתמש בtolerance גבוה יותר עבור השוואת CPU vs CUDA
    image_visual_similarity_asserts_for_gpu_output(expected_gaussian_blur_filter_img); //to fix the tolerance for GPU vs CPU comparison
}


TEST_F(PureCudaUtils, AdaptiveThresholdFilter) 
{ 
    ASSERT_EQ(input_image.type(), CV_8UC1); // BGR input
    output.create(input_image.rows, input_image.cols, CV_8UC1);

    pure_cuda_adaptive_threshold(input_image, output);

    image_save(debug_output / "pure_cuda_adaptive_threshold_filter.png", output);

    image_visual_similarity_asserts_for_gpu_output(expected_adaptive_threshold_filter_img); //to fix the tolerance for GPU vs CPU comparison
}
TEST_F(PureCudaUtils, ThresholdFilter) 
{ 

    ASSERT_EQ(input_image.type(), CV_8UC1); // BGR input
    output.create(input_image.rows, input_image.cols, CV_8UC1);
    image_save(debug_output / "pure_cuda_in_threshold_filter.png", input_image);
    pure_cuda_threshold_filter(input_image, output);

    image_save(debug_output / "pure_cuda_threshold_filter.png", output);
    image_visual_similarity_asserts_for_gpu_output(expected_threshold_filter_img);
}



// /* drawEquilateralTriangle ? tested with triangle tests */