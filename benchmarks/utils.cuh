#include <string>
#include <functional>
#include <iostream>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>



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



//auto end = std::chrono::high_resolution_clock::now();