
#include "utils.cuh"
#include <utils_cpp.h>
#include <utils_pure_cuda.cuh>
#include <utils_cuda_opencv.cuh>

namespace benchmark 
{




void threshold_filter(const cv::Mat& src, cv::Mat& dst) 
{
    ::threshold_filter(src, dst);
}

void adaptive_threshold_filter(const cv::Mat& src, cv::Mat& dst) 
{
    ::adaptive_threshold_filter(src, dst);
}

void gaussian_blur_filter(const cv::Mat& src, cv::Mat& dst) 
{
    ::gaussian_blur_filter(src, dst);
}


void gray_filter(const cv::Mat& src, cv::Mat& dst) 
{
    ::gray_filter(src, dst);
}





void opencv_cuda_threshold_filter(const cv::Mat& src, cv::Mat& dst) 
{
    const cv::cuda::GpuMat src_gpu(src);
    cv::cuda::GpuMat dst_gpu;
    ::cuda_threshold_filter(src_gpu, dst_gpu);
    dst_gpu.download(dst);
}

void opencv_cuda_adaptive_threshold_filter(const cv::Mat& src, cv::Mat& dst) 
{
    const cv::cuda::GpuMat src_gpu(src);
    cv::cuda::GpuMat dst_gpu;
    ::cuda_adaptive_threshold_filter(src_gpu, dst_gpu);
    dst_gpu.download(dst);
}

void opencv_cuda_gaussian_blur_filter(const cv::Mat& src, cv::Mat& dst) 
{
    const cv::cuda::GpuMat src_gpu(src);
    cv::cuda::GpuMat dst_gpu;
    ::cuda_gaussian_blur_filter(src_gpu, dst_gpu);
    dst_gpu.download(dst);
}


void opencv_cuda_gray_filter(const cv::Mat& src, cv::Mat& dst) 
{
    const cv::cuda::GpuMat src_gpu(src);
    cv::cuda::GpuMat dst_gpu;
    ::cuda_gray_filter(src_gpu, dst_gpu);
    dst_gpu.download(dst);
}




void pure_cuda_threshold_filter(
    const cv::Mat& src,
    cv::Mat& dst
)
{
    CV_Assert(src.type() == CV_8UC1); // BGR input

    dst.create(src.rows, src.cols, CV_8UC1);

    // PURE CUDA על buffers ישירות:
    // עובר stride עבור source ו-destination, plus default stream
    ::pure_cuda_threshold_filter(src, dst);
    
    std::cout << "CUDA threshold_filter: output " << dst.cols << "x" << dst.rows << " channels=" << dst.channels() << std::endl;
    // אין צורך ב-GPU mats כאן כי אנחנו עובדים עם pure CUDA
}


void pure_cuda_adaptive_threshold(
    const cv::Mat& src,
    cv::Mat& dst
)
{
    cv::cuda::GpuMat gpu_src(src);
    cv::cuda::GpuMat gpu_dst;
    CV_Assert(src.type() == CV_8UC1); // BGR input

    dst.create(src.rows, src.cols, CV_8UC1);

    // PURE CUDA על buffers ישירות:
    // עובר stride עבור source ו-destination, plus default stream
    ::pure_cuda_adaptive_threshold(src, dst);
    
    std::cout << "CUDA adaptive_threshold_filter: output " << dst.cols << "x" << dst.rows << " channels=" << dst.channels() << std::endl;
    // אין צורך ב-GPU mats כאן כי אנחנו עובדים עם pure CUDA
}


void pure_cuda_gaussian_blur_filter(
    const cv::Mat& src,
    cv::Mat& dst
)
{
    //upload_gaussian_kernel_5x5(); // Assuming this function uploads the Gaussian kernel to the GPU

    CV_Assert(src.type() == CV_8UC1); // BGR input

    dst.create(src.rows, src.cols, CV_8UC1);

    // PURE CUDA על buffers ישירות:
    // עובר stride עבור source ו-destination, plus default stream
    ::pure_cuda_gaussian_blur_filter(src, dst);
    
    std::cout << "CUDA gaussian_blur_filter: output " << dst.cols << "x" << dst.rows << " channels=" << dst.channels() << std::endl;
    // אין צורך ב-GPU mats כאן כי אנחנו עובדים עם pure CUDA
}


void pure_cuda_gray_filter(const cv::Mat& src, cv::Mat& dst) {
    CV_Assert(src.type() == CV_8UC3); // BGR input

    dst.create(src.rows, src.cols, CV_8UC1);

    // PURE CUDA על buffers ישירות:

    ::cuda_pure_gray_filter(src, dst);
    std::cout << "CUDA gray_filter: output " << dst.cols << "x" << dst.rows << " channels=" << dst.channels() << std::endl;
    // אין צורך ב-GPU mats כאן כי אנחנו עובדים עם pure CUDA

}





} // namespace benchmark





namespace profiling
{   
}



struct AlgorithmEntry {
    std::string name;
    std::function<void(const cv::Mat&, cv::Mat&)> func;
    std::function<double(const std::string&, cv::Mat&)> profile_func = nullptr; // יכול להיות nullptr
};



void benchmark_and_compare_logic(
    const cv::Mat& input,
    std::vector<AlgorithmEntry>& algorithms,
    const fs::path& output_dir,
    const std::string& bench_name
);


int main()
{
    fs::path dir = fs::path(__FILE__).parent_path();
    fs::path output_path = dir / "out";
    fs::path input_path = dir / "input";
    fs::path input_image = input_path / "input_image.png";
    if (!fs::exists(output_path)) fs::create_directories(output_path);
    // 3 Channel BGR
    cv::Mat input_color = cv::imread(input_image.string(), cv::IMREAD_COLOR);
    if (input_color.empty()) { std::cerr << RED << "✗ ERROR: Failed to read " << input_image << RESET << std::endl; return 1; }
    std::cout << "✓ Loaded image: " << input_color.cols << "x" << input_color.rows << " channels=" << input_color.channels() << std::endl;
    // 1 Channel Grayscale
    cv::Mat input = cv::imread(input_image.string(), cv::IMREAD_GRAYSCALE);
    if (input.empty()) { std::cerr << RED << "✗ ERROR: Failed to read " << input_image << RESET << std::endl; return 1; }
    std::cout << "✓ Loaded image: " << input.cols << "x" << input.rows << " channels=" << input.channels() << std::endl;

    auto algorithms_gray_filter = std::vector<AlgorithmEntry>
    {
        {"CPU", benchmark::gray_filter, nullptr},
        {"CUDA_OPENCV", benchmark::opencv_cuda_gray_filter, nullptr},
        {"PURE_CUDA", benchmark::pure_cuda_gray_filter, nullptr}
        
    };
    benchmark_and_compare_logic(input_color, algorithms_gray_filter, output_path, "Gray_Filter");

    auto algorithms_gaussian_blur = std::vector<AlgorithmEntry>
    {
        {"CPU", benchmark::gaussian_blur_filter, nullptr},
        {"CUDA_OPENCV", benchmark::opencv_cuda_gaussian_blur_filter, nullptr},
        {"PURE_CUDA", benchmark::pure_cuda_gaussian_blur_filter, nullptr}
    };
    benchmark_and_compare_logic(input, algorithms_gaussian_blur, output_path, "Gaussian_Blur_Filter");

    auto algorithms_adaptive_threshold = std::vector<AlgorithmEntry>
    {
        {"CPU", benchmark::adaptive_threshold_filter, nullptr},
        {"CUDA_OPENCV", benchmark::opencv_cuda_adaptive_threshold_filter, nullptr},
        {"PURE_CUDA", benchmark::pure_cuda_adaptive_threshold, nullptr}
    };
    benchmark_and_compare_logic(input, algorithms_adaptive_threshold, output_path, "Adaptive_Threshold");

    auto algorithms_threshold_filter = std::vector<AlgorithmEntry>
    {
        {"CPU", benchmark::threshold_filter, nullptr},
        {"CUDA_OPENCV", benchmark::opencv_cuda_threshold_filter, nullptr},
        {"PURE_CUDA", benchmark::pure_cuda_threshold_filter, nullptr}
    };
    benchmark_and_compare_logic(input, algorithms_threshold_filter, output_path, "Threshold_Filter");

    std::cout << GREEN << "✓ Done." << RESET << std::endl;
    return 0;
}



void benchmark_and_compare_logic(
    const cv::Mat& input,
    std::vector<AlgorithmEntry>& algorithms,
    const fs::path& output_dir,
    const std::string& bench_name
) {
    const fs::path bench_dir = output_dir / bench_name;

    if (!fs::exists(output_dir)) fs::create_directories(output_dir);
    if (!fs::exists(bench_dir)) fs::create_directories(bench_dir);

    std::vector<ImageData> results;
    // Profiling
    for (const auto& alg : algorithms) {
        ImageData res;
        res.name = alg.name;
        auto t0 = std::chrono::high_resolution_clock::now();
        alg.func(input, res.output_image);
        auto t1 = std::chrono::high_resolution_clock::now();
        res.duration = std::chrono::duration<double, std::milli>(t1-t0).count();
        res.proffiler_runs = 1; // Set to 1 for this run, can be adjusted if profiling is done multiple times
        res.output_image_path = bench_dir / ("output_" + alg.name + ".png");
        cv::imwrite(res.output_image_path, res.output_image);
        results.push_back(res);


        
    }

    // השוואה בין כל זוג אלגוריתמים
    fs::path bench_results_file = bench_dir / "bench_results.txt";
    std::filesystem::remove(bench_results_file);

    std::string banchmark_on = "";
    for (size_t i = 0; i < results.size(); ++i)
        banchmark_on += results[i].name + " ";

    title_print(bench_name, banchmark_on);

    for (size_t i = 0; i < results.size(); ++i) 
    {
        for (size_t j = i+1; j < results.size(); ++j) 
        {
            DiffStats diff = compute_image_difference(results[i].output_image, results[j].output_image);
            std::string pair_name = results[i].name + "_vs_" + results[j].name;
            diff.output_image_path = bench_dir / ("diff_" + pair_name + ".png");

            cv::imwrite(diff.output_image_path.string(), diff.diff_image);
            save_diff_stats_to_file(diff, pair_name, (bench_results_file).string());
            print_diff_stats(diff, pair_name, bench_name);

            std::cout << "➡️ " << results[i].name << " image: \n\t" << results[i].output_image_path.string() << std::endl;
            std::cout << "➡️ " << results[j].name << " image: \n\t" << results[j].output_image_path.string() << std::endl;
            std::cout <<  "➡️ Difference image: \n\t" << diff.output_image_path.string() << std::endl;
        }
    }
    for (const auto& res : results)
    {
        std::cout << BOLD << "⏱️ " << res.name << " duration: " << std::setprecision(3) << res.duration << " ms" << RESET << std::endl;
    }

    std::cout << BLUE << "=============================================" << RESET << std::endl;
    std::cout << std::endl;
}
