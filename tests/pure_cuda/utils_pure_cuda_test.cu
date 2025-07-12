#include <gtest/gtest.h>
#include <test_logic.h>  // #include <gtest/gtest.h> #include <opencv2/opencv.hpp> #include <filesystem> #include <string>
// the test functions
#include <utils_pure_cuda.cuh>

/* Cpp application*/
class CudaOpenCVUtils : public UtilsTest
{
    void SetUp() override
    {
        fs::create_directories(current_output);
        input_image = get_gray_filtered_img(input_gray_img);
        ASSERT_FALSE(input_image.empty()) << "Failed to load input image: " << input_gray_img;
        input_image_gpu.upload(input_image);
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
    cv::cuda::GpuMat input_image_gpu;
    cv::Mat output;
    cv::cuda::GpuMat output_gpu;
    
    void image_visual_similarity_asserts_for_gpu_output(const std::string& expected_path, double tol = 1e-6)
    {

        output_gpu.download(output);
        // Validate the expected image path first
        ASSERT_FALSE(expected_path.empty()) << "Expected gray filter image path is empty.";
        ASSERT_TRUE(fs::exists(expected_path)) << "Expected gray filter image does not exist: " + expected_path;
        
        cv::Mat expected_image = get_gray_filtered_img(expected_path);
        //ASSERT_FALSE(expected_image.empty()) << "Failed to load expected image: " << expected_path;
        //ASSERT_FALSE(result.empty()) << "Result image is empty.";
        TestLogic::image_visual_similarity_asserts(expected_image, output, tol);
    }



    cv::Mat get_gray_filtered_img(const std::string& input_gray_img) 
    {
        return cv::imread(input_gray_img, cv::IMREAD_GRAYSCALE);
    }
};


/*clean C++ Utils code Tests*/
//tested on gray image

TEST_F(CudaOpenCVUtils, CppCompiledUtilsMessage) 
{
    ASSERT_EQ(utils_print_message_pure_cuda(), "pure_cuda_utils"); 
}


TEST_F(CudaOpenCVUtils, GrayFilter) 
{ 
    cv::Mat input_image_color = cv::imread(original_img, cv::IMREAD_COLOR);
    cv::cuda::GpuMat input_image_gpu;
    
    input_image_gpu.upload(input_image_color);
    cuda_gray_filter(input_image_gpu, output_gpu);
    image_visual_similarity_asserts_for_gpu_output(expected_gray_filter_img);
}


TEST_F(CudaOpenCVUtils, GaussianBlueFilter) 
{ 

    cuda_gaussian_blur_filter(input_image_gpu, output_gpu);
    //output_gpu.download(output);
    //image_save(test_output / "output_gaussian_blur_filter999.png", output);

    // השתמש בtolerance גבוה יותר עבור השוואת CPU vs CUDA
    image_visual_similarity_asserts_for_gpu_output(expected_gaussian_blur_filter_img); //to fix the tolerance for GPU vs CPU comparison
}


TEST_F(CudaOpenCVUtils, AdaptiveThresholdFilter) 
{ 
    cuda_adaptive_threshold_filter(input_image_gpu, output_gpu);
    output_gpu.download(output);
    image_save(debug_output / "cuda_adaptive_threshold_filter99.png", output);

    image_visual_similarity_asserts_for_gpu_output(expected_adaptive_threshold_filter_img); //to fix the tolerance for GPU vs CPU comparison
}


TEST_F(CudaOpenCVUtils, ThresholdFilter) 
{ 
    cuda_threshold_filter(input_image_gpu, output_gpu);
    image_visual_similarity_asserts_for_gpu_output(expected_threshold_filter_img);
}


// /* drawEquilateralTriangle ? tested with triangle tests */