#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/quality/qualityssim.hpp>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <functional>
#include <fstream>
#include <numeric>


// Console coloring
#define BLACK   "\033[30m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"
#define RESET   "\033[0m"
#define BOLD    "\033[1m"
#define UNDERLINE "\033[4m"
#define REVERSED "\033[7m"
#define BG_BLACK   "\033[40m"
#define BG_RED     "\033[41m"
#define BG_GREEN   "\033[42m"
#define BG_YELLOW  "\033[43m"
#define BG_BLUE    "\033[44m"
#define BG_MAGENTA "\033[45m"
#define BG_CYAN    "\033[46m"
#define BG_WHITE   "\033[47m"


namespace fs = std::filesystem;

// ---------- Structs ----------
struct ImageData 
{

    std::string name;

    double duration = 0.0;

    cv::Mat output_image;

    int proffiler_runs = 0; 

    std::filesystem::path output_image_path;
};

struct DiffStats 
{
    
    double mean_diff = 0.0;
    double max_diff = 0.0;
    cv::Point max_diff_location;
    uint8_t pixel1_value = 0;
    uint8_t pixel2_value = 0;
    size_t different_pixels = 0;
    size_t total_pixels = 0;
    double norm_l2 = 0.0;
    double psnr = 0.0;
    double ssim = 0.0;
    cv::Mat diff_image;
    int width = 0;
    int height = 0;
    bool is_different_dimentions = false;
    bool is_different_types = false;
    double mean_time_taked = 0.0;
    int proffiler_runs = 0; //number of runnings
    fs::path output_image_path;
    std::string proffiler_function_name; //name of the profiled function
    std::shared_ptr<ImageData> image1 = nullptr; // Placeholder for ImageData reference, not used in this context
    std::shared_ptr<ImageData> image2 = nullptr; 
};

// ---------- Benchmark function ----------
double profile_function(
    const std::function<void(const cv::Mat&, cv::Mat&)>& fn,
    const cv::Mat& input,
    cv::Mat& output,
    int runs = 100)
{
    for (int i = 0; i < 2; ++i) fn(input, output); // warmup
    std::vector<double> times(runs);
    for (int i = 0; i < runs; ++i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        fn(input, output);
        // חובה ב־CUDA - ודא שהכל סיים לפני שמודדים
        cv::cuda::GpuMat().release();
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / runs;
    return mean;
}

void profile_and_update_stats(
    DiffStats& stats,
    const std::function<void(const cv::Mat&, cv::Mat&)>& fn,
    const cv::Mat& input,
    const std::string& function_name,
    int runs = 100)
{
    cv::Mat output;
    stats.mean_time_taked = profile_function(fn, input, output, runs);
    stats.proffiler_runs = runs;
    stats.proffiler_function_name = function_name;
}

// ---------- Diff logic ----------
DiffStats compute_image_difference(const cv::Mat& img1, const cv::Mat& img2) 
{
    DiffStats stats;
    if (img1.empty() || img2.empty()) return stats;
    stats.width = img1.cols;
    stats.height = img1.rows;

    if (img1.size() != img2.size()) stats.is_different_dimentions = true;
    if (img1.type() != img2.type()) stats.is_different_types = true;
    if (stats.is_different_dimentions || stats.is_different_types) return stats;

    cv::Mat diff_abs;
    cv::absdiff(img1, img2, diff_abs);
    stats.diff_image = diff_abs.clone();
    stats.mean_diff = cv::mean(diff_abs)[0];

    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(diff_abs, &min_val, &max_val, &min_loc, &max_loc);
    stats.max_diff = max_val;
    stats.max_diff_location = max_loc;
    if (img1.channels() == 1) {
        stats.pixel1_value = img1.at<uint8_t>(max_loc);
        stats.pixel2_value = img2.at<uint8_t>(max_loc);
    }
    cv::Mat non_zero_mask; 
    cv::threshold(diff_abs, non_zero_mask, 0, 255, cv::THRESH_BINARY);
    stats.different_pixels = cv::countNonZero(non_zero_mask);
    stats.total_pixels = img1.total();

    stats.norm_l2 = cv::norm(img1, img2, cv::NORM_L1);
    stats.psnr = cv::PSNR(img1, img2);
    stats.ssim = cv::quality::QualitySSIM::compute(img1, img2, cv::noArray())[0];
    return stats;
}


void title_print(const std::string& title, const std::string& banchmark_on ="") 
{
    std::cout << BOLD << YELLOW << "\n=== Benchmark: " << UNDERLINE <<  title << RESET << ": " << BOLD << YELLOW  << banchmark_on << RESET << BOLD << YELLOW << " === " << RESET << std::endl;
}   
// ---------- Colorful print ----------
void print_diff_stats(const DiffStats& stats, const std::string& test_name, const std::string& function_name = "") 
{
    bool good_ssim = stats.ssim >= 0.92;
    bool good_psnr = stats.psnr > 30.0;
    bool few_diff_pixels = stats.different_pixels < stats.total_pixels * 0.01;

    std::string ssim_col = good_ssim ? GREEN : (stats.ssim > 0.80 ? YELLOW : RED);
    std::string psnr_col = good_psnr ? GREEN : YELLOW;
    std::string diff_col = few_diff_pixels ? GREEN : YELLOW;
    std::string icon = good_ssim ? "✔️" : (stats.ssim > 0.80 ? "⚠️" : "❌");

    std::cout << BLUE << "\n=== " << UNDERLINE << test_name << RESET << BLUE << " Difference Analysis ===" << RESET << std::endl;
    if(!function_name.empty())
        std::cout << BOLD << CYAN << "= " << UNDERLINE << function_name << RESET << BOLD << CYAN  << " :" << RESET << std::endl;
    std::cout << "Image Size: " << stats.width << " x " << stats.height << std::endl;
    std::cout << "Mean difference per pixel: " << stats.mean_diff << std::endl;
    std::cout << "Max difference per pixel: " << stats.max_diff << std::endl;
    std::cout << "Max difference location: (" << stats.max_diff_location.x << ", " << stats.max_diff_location.y << ")" << std::endl;
    std::cout << "Pixel values at max diff: " << static_cast<int>(stats.pixel1_value) << " vs " << static_cast<int>(stats.pixel2_value) << std::endl;
    std::cout << diff_col << "Different pixels: " << stats.different_pixels << " / " << stats.total_pixels 
              << " (" << (100.0 * stats.different_pixels / stats.total_pixels) << "%)" << RESET << std::endl;
    std::cout << "Norm: " << stats.norm_l2  << std::endl;
    std::cout << psnr_col << "PSNR: " << stats.psnr << RESET << std::endl;
    std::cout << ssim_col << "SSIM: " << stats.ssim << " " << icon << RESET << std::endl;
    if(stats.proffiler_runs > 0)
    {
        std::cout << BOLD << CYAN << "⏱️ Mean Time (" << stats.proffiler_runs << " runs): " 
                  << std::setprecision(3) << stats.mean_time_taked << " ms" << RESET << std::endl;
    }
    else
    {
        std::cout << BOLD << BLUE << "⏱️ Mean Time: Not profiled" << RESET << std::endl;
    }
    //std::cout << BLUE << "=============================================" << RESET << std::endl;
}

// ---------- Save colorful diff to file (no ansi, but with icons and pretty layout) ----------
void save_diff_stats_to_file(const DiffStats& stats, const std::string& test_name, const std::string& filename, const std::string& function_name = "")
{
    std::ofstream out(filename, std::ios_base::app);
    bool good_ssim = stats.ssim >= 0.92;
    bool good_psnr = stats.psnr > 30.0;
    bool few_diff_pixels = stats.different_pixels < stats.total_pixels * 0.01;
    std::string icon = good_ssim ? "✔️" : (stats.ssim > 0.80 ? "⚠️" : "❌");

    out << "\n=== " << test_name << " Difference Analysis ===\n";
    if(!function_name.empty())
        out << "= " << function_name << " : =\n";
    out << "Image Size: " << stats.width << " x " << stats.height << "\n";
    out << "Mean difference per pixel: " << stats.mean_diff << "\n";
    out << "Max difference per pixel: " << stats.max_diff << "\n";
    out << "Max difference location: (" << stats.max_diff_location.x << ", " << stats.max_diff_location.y << ")\n";
    out << "Pixel values at max diff: " << static_cast<int>(stats.pixel1_value) << " vs " << static_cast<int>(stats.pixel2_value) << "\n";
    out << "Different pixels: " << stats.different_pixels << " / " << stats.total_pixels 
        << " (" << (100.0 * stats.different_pixels / stats.total_pixels) << "%)\n";
    out << "Norm: " << stats.norm_l2  << "\n";
    out << "PSNR: " << stats.psnr << "\n";
    out << "SSIM: " << stats.ssim << " " << icon << "\n";
    if(stats.proffiler_runs > 0)
    {
        out << "⏱️ Mean Time (" << stats.proffiler_runs << " runs): " 
            << std::setprecision(3) << stats.mean_time_taked << " ms\n";
    }
    else
    {
        out << "⏱️ Mean Time: Not profiled\n";
    }
    out << "=============================================\n" << std::endl;
    out.close();
}






void save_diff_stats_to_file2(const DiffStats& stats, const std::string& label, const std::string& filename)
{
    std::ofstream out(filename, std::ios_base::app);
    out << std::fixed << std::setprecision(6)
        << label << ","
        << stats.mean_diff << ","
        << stats.max_diff << ","
        << stats.different_pixels << ","
        << stats.total_pixels << ","
        << stats.norm_l2 << ","
        << stats.psnr << ","
        << stats.ssim << ","
        << stats.mean_time_taked << ","
        << stats.proffiler_runs << std::endl;
    out.close();
}