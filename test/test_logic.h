#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>


namespace fs = std::filesystem;

class TriangleImageTest : public ::testing::Test 
{
protected:
    fs::path dir = fs::path(__FILE__).parent_path();
    fs::path reference_input = dir / "reference_input";
    fs::path reference_output = dir / "reference_output";
    fs::path current_output = dir / "current_output";

    fs::path triangle_img = reference_input / "triangle.png";
    fs::path other_triangle_img = reference_input / "other_triangle.png";
    fs::path expected_triangle_img = reference_output / "triangle.png";
    fs::path expected_triangle_contours_img = reference_output / "triangle_contours.png";
    fs::path expected_other_triangle_contours_img = reference_output / "other_triangle_contours.png";
    fs::path new_triangle_img = current_output / "triangle.png";

    // Helper function for image similarity check
    void image_similarity_asserts(const cv::Mat& expected, const cv::Mat& result, double tol = 1e-6)
    {
        ASSERT_FALSE(expected.empty()) << "Reference image is missing!";
        ASSERT_FALSE(result.empty()) << "Result image is empty!";
        ASSERT_EQ(result.size(), expected.size()) << "Image sizes differ";
        ASSERT_EQ(result.type(), expected.type()) << "Image types differ";
        double diff = cv::norm(result, expected, cv::NORM_L2);
        ASSERT_LT(diff, tol) << "Images are not similar enough! Diff: " << diff;
    }

    void img_save_the_difference_between_images(fs::path&& diff_img_path, cv::Mat& expected, cv::Mat& result )
    {
        cv::Mat diff;
        cv::absdiff(expected, result, diff);
        cv::imwrite(diff_img_path, diff);
    }

};
