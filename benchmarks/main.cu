
#include "utils.cuh"
#include <utils_cpp.h>
#include <utils_cuda_opencv.cuh>


void adaptive_threshold_cpu(const cv::Mat& src, cv::Mat& dst) 
{
    
    adaptive_threshold_filter(src, dst);
}

void adaptive_threshold_cuda_opencv(const cv::Mat& src, cv::Mat& dst) {
    cv::cuda::GpuMat gpu_src(src);
    cv::cuda::GpuMat gpu_dst;
    cuda_adaptive_threshold_filter(gpu_src, gpu_dst);
    gpu_dst.download(dst);
}



int main()
{
    fs::path dir = fs::path(__FILE__).parent_path();
    fs::path output_path = dir / "out";
    fs::path input_path = dir / "input_image.png";

    // יצירת תיקיית פלט אם לא קיימת
    if (!fs::exists(output_path))
        fs::create_directories(output_path);

    cv::Mat input = cv::imread(input_path.string(), cv::IMREAD_GRAYSCALE);
    if (input.empty()) {
        std::cerr << RED << "✗ ERROR: Failed to read " << input_path << RESET << std::endl;
        return 1;
    }

    // יצירת פלט CPU ו־CUDA
    ImageData cpu = run_image_process("CPU", adaptive_threshold_cpu, input);
    ImageData cuda = run_image_process("CUDA_OPENCV", adaptive_threshold_cuda_opencv, input);

    // שמירת תמונות פלט
    cv::imwrite((output_path / "output_CPU.png").string(), cpu.output_image);
    cv::imwrite((output_path / "output_CUDA_OPENCV.png").string(), cuda.output_image);

    // חישוב diff
    DiffStats diff = compute_image_difference(cpu.output_image, cuda.output_image);

    // שמירת diff
    cv::imwrite((output_path / "diff_CPU__CUDA_OPENCV.png").string(), diff.diff_image);

    // הצגה והדפסה לקובץ
    print_diff_stats(diff, "CPU vs CUDA_OPENCV");
    save_diff_stats_to_file(diff, "CPU vs CUDA_OPENCV", (output_path / "bench_results.txt").string());

    // זמני ריצה (כללי)
    std::cout << BOLD << "⏱️ CPU duration: " << std::setprecision(3) << cpu.duration << " ms" << RESET << std::endl;
    std::cout << BOLD << "⏱️ CUDA_OPENCV duration: " << std::setprecision(3) << cuda.duration << " ms" << RESET << std::endl;

    std::cout << GREEN << "✓ Done." << RESET << std::endl;
    return 0;
}





/*

    fs::path dir = fs::path(__FILE__).parent_path();
    fs::path output_path = dir / "out";
    fs::path input_path = dir / "input_image.png";
    fs::path expected_cpu_path = dir / "utils_adaptive_threshold_filter.png";
    cv::Mat cpu_expected = cv::imread(expected_cpu_path.string(), cv::IMREAD_GRAYSCALE);

    cv::imwrite(output_path/"output_CUDA_OPENCV.png", cuda.output_image);
    
*/