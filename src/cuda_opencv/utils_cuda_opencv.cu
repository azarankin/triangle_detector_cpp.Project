#include <utils_cuda_opencv.cuh>
#include <debug.cuh>




std::string utils_print_message() 
{
    std::cout << "\t\t\t\t\t\t\tPrinting from CUDA OpenCV Utils library!" << std::endl;
    return "cuda_opencv_utils";
}


// תחליף את adaptiveThreshold על CPU בלוגיקה CUDA הזו:
void cuda_adaptive_threshold(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int blockSize, double C) 
{
    // (1) ממוצע לוקאלי – boxFilter CUDA
    cv::cuda::GpuMat mean_local;
    auto boxFilter = cv::cuda::createBoxFilter(src.type(), src.type(), cv::Size(blockSize, blockSize));
    boxFilter->apply(src, mean_local);

    // (2) מחסרים C מהמטריצה
    cv::cuda::GpuMat threshMat;
    cv::cuda::subtract(mean_local, cv::Scalar::all(C), threshMat);

    // (3) משווים: src > threshMat ? 255 : 0 (THRESH_BINARY)
    cv::cuda::compare(src, threshMat, dst, cv::CMP_GT); // dst = mask
    // dst יוצא בינארי (0/255)
}


void cuda_gray_filter(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst)
{
    cv::cuda::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
    SAVE_DEBUG_IMAGE(dst);
}


void cuda_gaussian_blur_filter(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst)
{
    auto gauss_filter = cv::cuda::createGaussianFilter(src.type(), src.type(), cv::Size(5, 5), 0);
    gauss_filter->apply(src, dst);
    SAVE_DEBUG_IMAGE(dst);
}


void cuda_adaptive_threshold_filter(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst)
{
    cuda_adaptive_threshold(src, dst, /*blockSize=*/11, /*C=*/5);
    SAVE_DEBUG_IMAGE(dst);
}


void cuda_threshold_filter(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst)
{
    cv::cuda::threshold(src, dst, 127, 255, cv::THRESH_BINARY);
    SAVE_DEBUG_IMAGE(dst);
}

