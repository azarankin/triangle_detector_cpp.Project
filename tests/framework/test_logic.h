#pragma once
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>


    //definition error check
#ifndef TEST_OUTPUT_DIRECTORY
    #define TEST_OUTPUT_DIRECTORY ""
    #error "TEST_OUTPUT_DIRECTORY is not defined! Please define it in your CMakeLists.txt."
#endif
#ifndef TEST_OUTPUT_ARCHIVE_DIRECTORY
    #define TEST_OUTPUT_ARCHIVE_DIRECTORY ""
    #error "TEST_OUTPUT_ARCHIVE_DIRECTORY is not defined! Please define it in your CMakeLists.txt."
#endif
#ifndef TEST_REFERENCE_INPUT_DIRECTORY
    #define TEST_REFERENCE_INPUT_DIRECTORY ""
    #error "TEST_OUTPUT_ARCHIVE_DIRECTORY is not defined! Please define it in your CMakeLists.txt."
#endif
#ifndef TEST_REFERENCE_OUTPUT_DIRECTORY
    #define TEST_REFERENCE_OUTPUT_DIRECTORY ""
    #error "TEST_OUTPUT_ARCHIVE_DIRECTORY is not defined! Please define it in your CMakeLists.txt."
#endif
#ifndef TEST_DEBUG_OUTPUT_DIRECTORY
    #define TEST_DEBUG_OUTPUT_DIRECTORY ""
    #error "TEST_DEBUG_OUTPUT_DIRECTORY is not defined! Please define it in your CMakeLists.txt."
#endif



// #pragma message("CMake: TEST_OUTPUT_DIRECTORY=" TEST_OUTPUT_DIRECTORY)
// #pragma message("CMake: TEST_DEBUG_OUTPUT_DIRECTORY=" TEST_DEBUG_OUTPUT_DIRECTORY)
// #pragma message("CMake: TEST_OUTPUT_ARCHIVE_DIRECTORY=" TEST_OUTPUT_ARCHIVE_DIRECTORY)
// #pragma message("CMake: TEST_REFERENCE_INPUT_DIRECTORY=" TEST_REFERENCE_INPUT_DIRECTORY)
// #pragma message("CMake: TEST_REFERENCE_OUTPUT_DIRECTORY=" TEST_REFERENCE_OUTPUT_DIRECTORY)


namespace fs = std::filesystem;

class TestLogic : public ::testing::Test 
{
protected:
    fs::path dir = fs::path(__FILE__).parent_path().parent_path() / "data"; // Assuming the data directory is two levels up from the current file
    fs::path reference_input = !std::string(TEST_REFERENCE_INPUT_DIRECTORY).empty()
        ? fs::path(TEST_REFERENCE_INPUT_DIRECTORY)
        : dir / "reference_input";
    fs::path debug_output = !std::string(TEST_DEBUG_OUTPUT_DIRECTORY).empty()
        ? fs::path(TEST_DEBUG_OUTPUT_DIRECTORY)
        : current_output / "test_debug_output";
    fs::path reference_output = !std::string(TEST_REFERENCE_OUTPUT_DIRECTORY).empty()
        ? fs::path(TEST_REFERENCE_OUTPUT_DIRECTORY)
        : dir / "reference_output";
    fs::path current_output = !std::string(TEST_OUTPUT_DIRECTORY).empty()
        ? fs::path(TEST_OUTPUT_DIRECTORY)
        : dir / "current_output";
    fs::path archive_current_output = !std::string(TEST_OUTPUT_ARCHIVE_DIRECTORY).empty()
        ? fs::path(TEST_OUTPUT_ARCHIVE_DIRECTORY)
        : dir / "archive_current_output";  
        

    // Helper function for image similarity check
    void image_visual_similarity_asserts(const cv::Mat& expected, const cv::Mat& result, double tol = 0.96);
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
    fs::path expected_gray_filter_img = reference_output / "utils_gray_filter.png";
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
