#include <shape_detector_common.h>
#include <utils_cuda_opencv.cuh>


#include <cuda_runtime.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp> // cv::cuda::cvtColor
#include <opencv2/cudafilters.hpp>

std::string triangle_print_message() 
{
    std::cout << "\t\t\t\t\t\t\tPrinting from CUDA OpenCV Triangel Detector library!" << std::endl;
    return "cuda_opencv_triangle_detector";
}


std::vector<cv::Point> find_shape_contour(const std::string& address)
{
    // Load image
    cv::Mat host_img = cv::imread(address);
    if (host_img.empty())
        throw std::runtime_error("Failed to load image: " + address);
    
    //std::cout << "DEBUG: Loaded image: " << host_img.cols << "x" << host_img.rows << " channels=" << host_img.channels() << std::endl;

    // Upload to GPU
    cv::cuda::GpuMat gpu_img(host_img);
    //std::cout << "DEBUG: GPU upload: " << gpu_img.cols << "x" << gpu_img.rows << " channels=" << gpu_img.channels() << std::endl;

    // Convert to grayscale
    cv::cuda::GpuMat gpu_gray;
    cuda_gray_filter(gpu_img ,gpu_gray); // , "find_shape_contour__gray"
    //std::cout << "DEBUG: After gray: " << gpu_gray.cols << "x" << gpu_gray.rows << " channels=" << gpu_gray.channels() << std::endl;

    // Gaussian blur
    cv::cuda::GpuMat gpu_blur;
    cuda_gaussian_blur_filter(gpu_gray, gpu_blur); // , "find_shape_contour__blur"
    //std::cout << "DEBUG: After blur: " << gpu_blur.cols << "x" << gpu_blur.rows << " channels=" << gpu_blur.channels() << std::endl;

    // Threshold
    cv::cuda::GpuMat gpu_thresh;
    cv::Mat thresh1;
    cuda_threshold_filter(gpu_blur, gpu_thresh); // , "find_shape_contour__thresh"
    //std::cout << "DEBUG: After threshold GPU: " << gpu_thresh.cols << "x" << gpu_thresh.rows << " channels=" << gpu_thresh.channels() << std::endl;
    
    gpu_thresh.download(thresh1);
    //std::cout << "DEBUG: After download: " << thresh1.cols << "x" << thresh1.rows << " channels=" << thresh1.channels() << std::endl;

    // Find contours (CPU only)
    std::vector<std::vector<cv::Point>> contours_template;
    find_contours(contours_template, thresh1);


    // Sort contours by area
    auto second_largest_contour = get_second_largest_contour(contours_template);

    // Draw the second largest contour
    return second_largest_contour;
}


std::vector<std::vector<cv::Point>> contour_compare(
    cv::Mat& target_frame, 
    const std::vector<cv::Point>& template_contour,
    double match_threshold, 
    double min_area_ratio, 
    double max_area_ratio)
{
    //std::cout << "DEBUG: contour_compare input: " << target_frame.cols << "x" << target_frame.rows << " channels=" << target_frame.channels() << std::endl;
    
    // Upload image to GPU
    cv::cuda::GpuMat gpu_frame(target_frame);
    //std::cout << "DEBUG: contour_compare GPU upload: " << gpu_frame.cols << "x" << gpu_frame.rows << " channels=" << gpu_frame.channels() << std::endl;

    // Gray filter (CUDA)
    cv::cuda::GpuMat gpu_gray;
    cuda_gray_filter(gpu_frame ,gpu_gray); // , "contour_compare__gray"
    //std::cout << "DEBUG: contour_compare after gray: " << gpu_gray.cols << "x" << gpu_gray.rows << " channels=" << gpu_gray.channels() << std::endl;

    // Gaussian blur (CUDA)
    cv::cuda::GpuMat gpu_blur;
    cuda_gaussian_blur_filter(gpu_gray, gpu_blur); // , "contour_compare__blur"
    //std::cout << "DEBUG: contour_compare after blur: " << gpu_blur.cols << "x" << gpu_blur.rows << " channels=" << gpu_blur.channels() << std::endl;

    // Adaptive threshold (CUDA)
    cv::cuda::GpuMat gpu_thresh;
    cuda_adaptive_threshold_filter(gpu_blur, gpu_thresh); // , "contour_compare__thresh"
    //std::cout << "DEBUG: contour_compare after adaptive threshold: " << gpu_thresh.cols << "x" << gpu_thresh.rows << " channels=" << gpu_thresh.channels() << std::endl;

    // Download the thresholded image back to CPU
    cv::Mat thresh2;
    gpu_thresh.download(thresh2);
    //std::cout << "DEBUG: contour_compare after download: " << thresh2.cols << "x" << thresh2.rows << " channels=" << thresh2.channels() << std::endl;

    // Contours (CPU)
    std::vector<std::vector<cv::Point>> contours_target;
    find_contours(contours_target, thresh2);

    std::vector<std::vector<cv::Point>> closest_contours = find_closest_contours(target_frame, template_contour, contours_target, match_threshold, min_area_ratio, max_area_ratio);

    return closest_contours;
}


void create_triangle_image(const std::string& filename, cv::Size size, int margin) 
{
    int width = size.width;
    int height = size.height;

    uchar3* d_img;
    size_t imgSize = width * height * sizeof(uchar3);
    cudaMalloc(&d_img, imgSize);
    
    dim3 block(256);  // כל בלוק ממלא שורות (thread לכל y)
    dim3 grid((height + block.x - 1) / block.x);
    drawEquilateralTriangleKernel<<<grid, block>>>(d_img, width, height, margin);

    cudaDeviceSynchronize();

    // העברה חזרה ל־CPU וכתיבה
    std::vector<uchar3> h_img(width * height);
    cudaMemcpy(h_img.data(), d_img, imgSize, cudaMemcpyDeviceToHost);
    cudaFree(d_img);

    // עטיפה לתוך cv::Mat
    cv::Mat img(height, width, CV_8UC3, h_img.data());
    cv::imwrite(filename, img);

    std::cout << "Triangle image (CUDA) saved as " << filename << std::endl;
}


