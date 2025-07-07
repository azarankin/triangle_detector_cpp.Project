#include <debug.h>

void save_debug_image(const std::string& name, const cv::Mat& img)
{

//if(!g_debug_config.enable)
//    return;
if (img.empty()) return;
//td::filesystem::create_directories(out_dir);
cv::Mat to_save;
if (img.channels() == 4)
    cv::cvtColor(img, to_save, cv::COLOR_BGRA2RGBA);
else if (img.channels() == 3)
    cv::cvtColor(img, to_save, cv::COLOR_BGR2RGB);
else
    to_save = img;

//cv::imwrite(g_debug_config.output_dir + (!g_debug_config.prefix.empty() ? ("/" + g_debug_config.prefix +  "__") : "/") + name + ".png", to_save);

}


