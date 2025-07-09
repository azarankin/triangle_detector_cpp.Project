#include <debug.h>
#include <filesystem>
#include <string>

#ifndef DEBUG_IMAGE_DIR
#define DEBUG_IMAGE_DIR ""
#endif




void save_debug_image(const std::string& name, const cv::Mat& img)
{
    if (img.empty()) return;

    std::filesystem::create_directories(std::string(DEBUG_IMAGE_DIR));
    cv::Mat to_save;
    if (img.channels() == 4)
        cv::cvtColor(img, to_save, cv::COLOR_BGRA2RGBA);
    else if (img.channels() == 3)
        cv::cvtColor(img, to_save, cv::COLOR_BGR2RGB);
    else
        to_save = img;

    cv::imwrite(std::string(DEBUG_IMAGE_DIR) + "/" + name + ".png", to_save);
}


