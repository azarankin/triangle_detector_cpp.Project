#include <debug.h>
#include <filesystem>
#include <string>

#ifndef DEBUG_IMAGE_DIR
#define DEBUG_IMAGE_DIR ""
#endif
#ifndef DEBUG_IMAGE_DEFAULT_DIR
#define DEBUG_IMAGE_DEFAULT_DIR ""
#endif


void save_debug_image(const std::string& name, const cv::Mat& img)
{
    if (img.empty()) return;
    std::filesystem::create_directories(std::string(DEBUG_IMAGE_DIR));
    std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 0};
    cv::imwrite(std::string(DEBUG_IMAGE_DIR) + "/" + name + ".png", img, params);
}


