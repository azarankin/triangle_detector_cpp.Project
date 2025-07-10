#pragma once
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>



#ifndef TEST_OUTPUT_DIRECTORY
    //definition error check
    //#error "TEST_OUTPUT_DIRECTORY is not defined! Please define it in your CMakeLists.txt."
    #define TEST_OUTPUT_DIRECTORY ""
#endif
#ifndef TEST_OUTPUT_ARCHIVE_DIRECTORY
    #define TEST_OUTPUT_ARCHIVE_DIRECTORY ""
#endif

#pragma message("CMake: TEST_OUTPUT_DIRECTORY=" TEST_OUTPUT_DIRECTORY)
#pragma message("CMake: TEST_OUTPUT_ARCHIVE_DIRECTORY=" TEST_OUTPUT_ARCHIVE_DIRECTORY)



namespace fs = std::filesystem;

class TestLogic : public ::testing::Test 
{
protected:
    fs::path dir = fs::path(__FILE__).parent_path();
    fs::path reference_input = dir / "reference_input";
    fs::path reference_output = dir / "reference_output";
    fs::path current_output = std::string(TEST_OUTPUT_DIRECTORY).empty()
        ? dir / "current_output"
        : fs::path(TEST_OUTPUT_DIRECTORY);
    fs::path archive_current_output = std::string(TEST_OUTPUT_ARCHIVE_DIRECTORY).empty()
        ? dir / "archive_current_output"
        : fs::path(TEST_OUTPUT_ARCHIVE_DIRECTORY);
    fs::path test_output = current_output / "test_output";
    // Helper function for image similarity check
    void image_similarity_asserts(const cv::Mat& expected, const cv::Mat& result, double tol = 1e-6);
    void img_save_the_difference_between_images(const fs::path& diff_img_path, cv::Mat& expected, cv::Mat& result );
    void clear_directory(const fs::path& dir_path); 
    std::string sanitize_filename(const std::string& name);
    void archive_directory(const fs::path& src, const fs::path& archive_dir);
    std::string get_current_test_name();

    void image_save(const fs::path& path, const cv::Mat& img);


};

class UtilsTest : public TestLogic
{
protected:
    fs::path original_img = reference_input / "utils_original_image.png";
    fs::path input_gray_img = reference_input / "utils_gray_filter.png";
    fs::path expected_gray_filter_img = reference_output / "utils_gray_filter.png";;
    fs::path expected_gaussian_blur_filter_img = reference_output / "utils_gaussian_blur_filter.png";
    fs::path expected_adaptive_threshold_filter_img = reference_output / "utils_adaptive_threshold_filter.png";
    fs::path expected_threshold_filter_img = reference_output / "utils_threshold_filter.png";
    
    // Add test-specific members here if needed
};

class TriangleImageTest : public TestLogic
{
protected:
    fs::path triangle_img = reference_input / "triangle.png";
    fs::path other_triangle_img = reference_input / "other_triangle.png";
    fs::path expected_triangle_img = reference_output / "triangle.png";
    fs::path expected_triangle_contours_img = reference_output / "triangle_contours.png";
    fs::path expected_other_triangle_contours_img = reference_output / "other_triangle_contours.png";
    fs::path new_triangle_img = current_output / "triangle.png";
};
