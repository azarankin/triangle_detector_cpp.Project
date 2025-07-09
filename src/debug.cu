#include <debug.cuh>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>

void save_debug_image(const std::string& name, const cv::cuda::GpuMat& gpu_img)
{
    cv::Mat cpu_img;
    gpu_img.download(cpu_img);
    save_debug_image(name, cpu_img);
}


