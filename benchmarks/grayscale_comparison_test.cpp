#include <gtest/gtest.h>
#include "test_logic.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <filesystem>
#include <iomanip>

namespace fs = std::filesystem;

class GrayscaleComparisonTest : public TestLogic
{
protected:
    void SetUp() override
    {
        fs::create_directories(current_output);
    }

    void TearDown() override 
    {
        std::string test_name = get_current_test_name();
        archive_directory(current_output, archive_current_output, test_name);
    }

    struct DiffStats {
        double mean_diff = 0.0;
        double max_diff = 0.0;
        cv::Point max_diff_location;
        uint8_t pixel1_value = 0;
        uint8_t pixel2_value = 0;
        size_t different_pixels = 0;
        size_t total_pixels = 0;
    };

    DiffStats compute_image_difference(const cv::Mat& img1, const cv::Mat& img2, 
                                     const std::string& diff_output_path = "") 
    {
        DiffStats stats;
        
        // Ensure images have same dimensions and type
        EXPECT_EQ(img1.size(), img2.size()) << "Images have different sizes";
        EXPECT_EQ(img1.type(), img2.type()) << "Images have different types";
        
        if (img1.size() != img2.size() || img1.type() != img2.type()) {
            return stats; // Return empty stats on mismatch
        }

        // Compute absolute difference
        cv::Mat diff_abs;
        cv::absdiff(img1, img2, diff_abs);
        
        // Calculate statistics
        cv::Scalar mean_scalar = cv::mean(diff_abs);
        stats.mean_diff = mean_scalar[0];
        
        double min_val, max_val;
        cv::Point min_loc, max_loc;
        cv::minMaxLoc(diff_abs, &min_val, &max_val, &min_loc, &max_loc);
        stats.max_diff = max_val;
        stats.max_diff_location = max_loc;
        
        // Get pixel values at max difference location
        if (img1.channels() == 1) {
            stats.pixel1_value = img1.at<uint8_t>(max_loc);
            stats.pixel2_value = img2.at<uint8_t>(max_loc);
        }
        
        // Count different pixels
        cv::Mat non_zero_mask;
        cv::threshold(diff_abs, non_zero_mask, 0, 255, cv::THRESH_BINARY);
        stats.different_pixels = cv::countNonZero(non_zero_mask);
        stats.total_pixels = img1.total();
        
        // Save difference image if path provided
        if (!diff_output_path.empty()) {
            // Enhance visibility of differences
            cv::Mat diff_enhanced;
            diff_abs.convertTo(diff_enhanced, CV_8U);
            
            // Scale up small differences for visibility
            cv::Mat diff_scaled;
            diff_enhanced.convertTo(diff_scaled, CV_8U, 10.0, 0); // Scale by 10x
            
            cv::imwrite(diff_output_path, diff_scaled);
            
            // Also save the original diff
            std::string raw_diff_path = diff_output_path;
            size_t dot_pos = raw_diff_path.find_last_of('.');
            if (dot_pos != std::string::npos) {
                raw_diff_path.insert(dot_pos, "_raw");
            }
            cv::imwrite(raw_diff_path, diff_abs);
        }
        
        return stats;
    }

    void print_diff_stats(const DiffStats& stats, const std::string& test_name) 
    {
        std::cout << "\n=== " << test_name << " Difference Analysis ===" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Mean difference per pixel: " << stats.mean_diff << std::endl;
        std::cout << "Max difference per pixel: " << stats.max_diff << std::endl;
        std::cout << "Max difference location: (" << stats.max_diff_location.x 
                  << ", " << stats.max_diff_location.y << ")" << std::endl;
        std::cout << "Pixel values at max diff: " << (int)stats.pixel1_value 
                  << " vs " << (int)stats.pixel2_value << std::endl;
        std::cout << "Different pixels: " << stats.different_pixels 
                  << " / " << stats.total_pixels 
                  << " (" << (100.0 * stats.different_pixels / stats.total_pixels) << "%)" << std::endl;
        std::cout << "=============================================\n" << std::endl;
    }
};

TEST_F(GrayscaleComparisonTest, CompareCpuVsCudaGrayscale)
{
    // Paths to the generated images
    fs::path cpu_image_path = current_output / "utils_gray_filter.png";
    fs::path cuda_image_path = current_output / "cuda_gray_filter_result.png";
    fs::path diff_image_path = current_output / "diff_gray_filter.png";
    
    // Check if both images exist
    ASSERT_TRUE(fs::exists(cpu_image_path)) 
        << "CPU grayscale image not found: " << cpu_image_path;
    ASSERT_TRUE(fs::exists(cuda_image_path)) 
        << "CUDA grayscale image not found: " << cuda_image_path;
    
    // Load both images in grayscale mode
    cv::Mat cpu_img = cv::imread(cpu_image_path.string(), cv::IMREAD_GRAYSCALE);
    cv::Mat cuda_img = cv::imread(cuda_image_path.string(), cv::IMREAD_GRAYSCALE);
    
    ASSERT_FALSE(cpu_img.empty()) << "Failed to load CPU image: " << cpu_image_path;
    ASSERT_FALSE(cuda_img.empty()) << "Failed to load CUDA image: " << cuda_image_path;
    
    // Compute differences
    DiffStats stats = compute_image_difference(cpu_img, cuda_img, diff_image_path.string());
    
    // Print detailed statistics
    print_diff_stats(stats, "CPU vs CUDA Grayscale");
    
    // Define tolerance levels
    const double MAX_MEAN_DIFF_TOLERANCE = 1.0;  // Max acceptable mean difference
    const double MAX_PIXEL_DIFF_TOLERANCE = 3.0; // Max acceptable single pixel difference
    const double MAX_DIFFERENT_PIXELS_PERCENT = 1.0; // Max % of different pixels
    
    // Assert the differences are within tolerance
    EXPECT_LE(stats.mean_diff, MAX_MEAN_DIFF_TOLERANCE) 
        << "Mean difference exceeds tolerance";
    EXPECT_LE(stats.max_diff, MAX_PIXEL_DIFF_TOLERANCE) 
        << "Maximum pixel difference exceeds tolerance";
    
    double different_pixels_percent = 100.0 * stats.different_pixels / stats.total_pixels;
    EXPECT_LE(different_pixels_percent, MAX_DIFFERENT_PIXELS_PERCENT) 
        << "Too many pixels differ between CPU and CUDA implementations";
    
    // For truly identical implementations, we might expect perfect matches
    if (stats.max_diff == 0.0) {
        std::cout << "✅ PERFECT MATCH: CPU and CUDA implementations are identical!" << std::endl;
    } else if (stats.max_diff <= 1.0) {
        std::cout << "✅ EXCELLENT: CPU and CUDA implementations are nearly identical!" << std::endl;
    } else if (stats.max_diff <= 3.0) {
        std::cout << "⚠️  GOOD: CPU and CUDA implementations have minor differences." << std::endl;
    } else {
        std::cout << "❌ WARNING: Significant differences detected between implementations!" << std::endl;
    }
}

// Alternative test that creates its own test images for comparison
TEST_F(GrayscaleComparisonTest, GenerateAndCompareGrayscaleConversions)
{
    // Load a test image
    fs::path test_input = reference_input / "triangle.png";
    ASSERT_TRUE(fs::exists(test_input)) << "Test input image not found: " << test_input;
    
    cv::Mat original = cv::imread(test_input.string(), cv::IMREAD_COLOR);
    ASSERT_FALSE(original.empty()) << "Failed to load test image";
    
    // Apply CPU grayscale conversion
    cv::Mat cpu_gray;
    cv::cvtColor(original, cpu_gray, cv::COLOR_BGR2GRAY);
    
    // Apply CUDA grayscale conversion
    cv::cuda::GpuMat gpu_original(original);
    cv::cuda::GpuMat gpu_gray;
    cv::cuda::cvtColor(gpu_original, gpu_gray, cv::COLOR_BGR2GRAY);
    
    cv::Mat cuda_gray;
    gpu_gray.download(cuda_gray);
    
    // Save both results
    cv::imwrite((current_output / "cpu_grayscale_direct.png").string(), cpu_gray);
    cv::imwrite((current_output / "cuda_grayscale_direct.png").string(), cuda_gray);
    
    // Compare the results
    fs::path diff_path = current_output / "diff_grayscale_direct.png";
    DiffStats stats = compute_image_difference(cpu_gray, cuda_gray, diff_path.string());
    
    print_diff_stats(stats, "Direct CPU vs CUDA cvtColor");
    
    // OpenCV's cv::cvtColor should be identical between CPU and CUDA
    EXPECT_EQ(stats.max_diff, 0.0) 
        << "OpenCV CPU and CUDA cvtColor should produce identical results";
}
