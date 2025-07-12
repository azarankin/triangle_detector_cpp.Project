#include <test_logic.h>
#include <opencv2/quality/qualityssim.hpp>

void TestLogic::image_visual_similarity_asserts(const cv::Mat& expected, const cv::Mat& result, double tol)
{
    ASSERT_FALSE(expected.empty()) << "Reference image is missing!";
    ASSERT_FALSE(result.empty()) << "Result image is empty!";
    ASSERT_EQ(result.size(), expected.size()) << "Image sizes differ";
    ASSERT_EQ(result.type(), expected.type()) << "Image types differ";
    //double diff = cv::norm(result, expected, cv::NORM_L2); //overkill for visual similarity
    //ASSERT_LT(diff, tol) << "Images are not similar enough! Diff: " << diff;
    double ssim = cv::quality::QualitySSIM::compute(expected, result, cv::noArray())[0];
    ASSERT_GT(ssim, tol) << "Images are not similar enough! SSIM: " << ssim;
}



void TestLogic::img_save_the_difference_between_images(const fs::path& diff_img_path, cv::Mat& expected, cv::Mat& result)
{
    cv::Mat diff;
    cv::absdiff(expected, result, diff);
    cv::imwrite(diff_img_path, diff);
}

void TestLogic::clear_directory(const fs::path& dir_path)
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

std::string TestLogic::sanitize_filename(const std::string& name) {
    std::string s = name;
    for (char& c : s) {
        if (!isalnum(c) && c != '_' && c != '-') c = '_';
    }
    return s;
}
    
    
void TestLogic::archive_directory(const fs::path& src, const fs::path& archive_dir) 
{
    if (!fs::exists(src) || fs::is_empty(src)) return;

    fs::create_directories(archive_dir);

    for (const auto& entry : fs::recursive_directory_iterator(src)) {
        fs::path rel_path = fs::relative(entry.path(), src);
        fs::path dst_path = archive_dir / rel_path;

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



std::string TestLogic::get_current_test_name() 
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

void TestLogic::image_save(const fs::path& path, const cv::Mat& img)
{
    if (path.empty()) return;
    if (img.empty()) return;

    std::filesystem::create_directories(path.parent_path());
    //std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 0};
    cv::imwrite(path, img);
}