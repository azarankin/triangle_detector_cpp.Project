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

class TriangleImageTest : public ::testing::Test 
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

    void img_save_the_difference_between_images(const fs::path& diff_img_path, cv::Mat& expected, cv::Mat& result )
    {
        cv::Mat diff;
        cv::absdiff(expected, result, diff);
        cv::imwrite(diff_img_path, diff);
    }

    void clear_directory(const fs::path& dir_path)
     {
        if (fs::exists(dir_path)) 
        {
            fs::remove_all(dir_path);
            fs::create_directory(dir_path); // צור מחדש כדי שיהיה מוכן לפלט הבא
        }
        else 
        {
            fs::create_directory(dir_path);
        }
    }

    std::string sanitize_filename(const std::string& name) {
        std::string s = name;
        for (char& c : s) {
            if (!isalnum(c) && c != '_' && c != '-') c = '_';
        }
        return s;
    }
     
    


 
void archive_directory(const fs::path& src, const fs::path& archive_dir, const std::string& test_name = "") {
    if (!fs::exists(src) || fs::is_empty(src)) return;

    fs::path final_archive_dir = test_name.empty() ? archive_dir : archive_dir / test_name;
    
    if (fs::exists(final_archive_dir)) {
        fs::remove_all(final_archive_dir);
    }
    fs::create_directories(final_archive_dir);

    for (const auto& entry : fs::recursive_directory_iterator(src)) {
        // הנתיב היחסי מהשורש src
        fs::path rel_path = fs::relative(entry.path(), src);
        fs::path dst_path = final_archive_dir / rel_path;

        std::cout << "Copying: " << entry.path() << " -> " << dst_path << std::endl;

        if (fs::is_directory(entry)) {
            fs::create_directories(dst_path);
        } else if (fs::is_regular_file(entry)) {
            fs::create_directories(dst_path.parent_path());
            fs::copy_file(entry.path(), dst_path, fs::copy_options::overwrite_existing);
        }
    }

    fs::remove_all(src);
    fs::create_directories(src);
}






    std::string get_current_test_name() 
    {
    //  GoogleTest 1.7
    const ::testing::TestInfo* const test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    if (test_info) {
        std::string case_name = test_info->test_case_name(); // או test_suite_name() בגוגל טסטים חדשים
        std::string test_name = test_info->name();
        return case_name + "_" + test_name;
    }
    return "UnknownTest";
}

};


