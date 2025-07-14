#include <shape_detector_common.h>
#include <utils_pure_cuda.cuh>


#include <cuda_runtime.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp> // cv::cuda::cvtColor
#include <opencv2/cudafilters.hpp>


std::string triangle_print_message() 
{
    std::cout << "\t\t\t\t\t\t\tPrinting from CUDA OpenCV Triangel Detector library!" << std::endl;
    return "pure_cuda_triangle_detector";
}


// אם אתה רוצה לעבור מבנה Mat → device array:
static void upload_mat_bgr_to_uchar3(const cv::Mat& mat, uchar3*& d_img, int& width, int& height) {
    if (mat.channels() != 3) throw std::runtime_error("Image is not BGR!");
    width = mat.cols;
    height = mat.rows;
    size_t bytes = width * height * sizeof(uchar3);
    cudaMalloc(&d_img, bytes);
    cudaMemcpy(d_img, mat.ptr<uchar3>(), bytes, cudaMemcpyHostToDevice);
}


// פונקציה עיקרית:
std::vector<cv::Point> find_shape_contour(const std::string& address)
{
    // --- 1. Load & upload image
    cv::Mat host_img = cv::imread(address, cv::IMREAD_COLOR);
    if (host_img.empty())
        throw std::runtime_error("Failed to load image: " + address);

    int width = host_img.cols, height = host_img.rows;

    // allocate and upload BGR uchar3
    uchar3* d_bgr = nullptr;
    cudaMalloc(&d_bgr, width * height * sizeof(uchar3));
    cudaMemcpy(d_bgr, host_img.ptr<uchar3>(), width * height * sizeof(uchar3), cudaMemcpyHostToDevice);

    // allocate intermediate device buffers
    unsigned char *d_gray = nullptr, *d_blur = nullptr, *d_thresh = nullptr;
    cudaMalloc(&d_gray, width * height);
    cudaMalloc(&d_blur, width * height);
    cudaMalloc(&d_thresh, width * height);

    cudaStream_t stream = 0;

    // --- 2. BGR2Gray kernel
    {
        dim3 block(16,16);
        dim3 grid((width+15)/16, (height+15)/16);
        bgr2gray_kernel<<<grid, block, 0, stream>>>(d_bgr, d_gray, width, height);
    }

    // --- 3. Gaussian blur kernel
    //upload_gaussian_kernel_5x5();
    {
        dim3 block(16,16);
        dim3 grid((width+15)/16, (height+15)/16);
        // שים לב שהקרנל שלך צריך src_stride/dst_stride (בד"כ width):
        gaussian_blur_kernel<<<grid, block, 0, stream>>>(
            d_gray, d_blur, width, height, width, width
        );
    }

    // --- 4. Threshold kernel
    {
        dim3 block(16,16);
        dim3 grid((width+15)/16, (height+15)/16);
        threshold_kernel<<<grid, block, 0, stream>>>(
            d_blur, d_thresh, width, height, width, width, 127, 255
        );
    }

    // --- 5. Download to host
    cv::Mat thresh_host(height, width, CV_8UC1);
    cudaMemcpy(thresh_host.data, d_thresh, width * height, cudaMemcpyDeviceToHost);

    // --- 6. Find contours (CPU only)
    std::vector<std::vector<cv::Point>> contours_template;
    find_contours(contours_template, thresh_host);
    auto second_largest_contour = get_second_largest_contour(contours_template);

    // --- 7. Cleanup
    cudaFree(d_bgr); cudaFree(d_gray); cudaFree(d_blur); cudaFree(d_thresh);

    return second_largest_contour;
}

std::vector<std::vector<cv::Point>> contour_compare(
    cv::Mat& target_frame, 
    const std::vector<cv::Point>& template_contour,
    double match_threshold, 
    double min_area_ratio, 
    double max_area_ratio)
{
    // 1. upload BGR to device
    int width = target_frame.cols, height = target_frame.rows;
    uchar3* d_bgr = nullptr;
    cudaMalloc(&d_bgr, width * height * sizeof(uchar3));
    cudaMemcpy(d_bgr, target_frame.ptr<uchar3>(), width * height * sizeof(uchar3), cudaMemcpyHostToDevice);

    // 2. allocate intermediates
    unsigned char* d_gray = nullptr;
    unsigned char* d_blur = nullptr;
    unsigned char* d_thresh = nullptr;
    cudaMalloc(&d_gray, width * height);
    cudaMalloc(&d_blur, width * height);
    cudaMalloc(&d_thresh, width * height);

    cudaStream_t stream = 0; // default stream

    // --- Gray filter (קרנל ישיר)
    {
        dim3 block(16, 16), grid((width+15)/16, (height+15)/16);
        bgr2gray_kernel<<<grid, block, 0, stream>>>(d_bgr, d_gray, width, height);
    }

    // --- Gaussian blur (קרנל ישיר)
    //upload_gaussian_kernel_5x5(); // מעלה ל־constant memory
    {
        dim3 block(16,16), grid((width+15)/16, (height+15)/16);
        gaussian_blur_kernel<<<grid, block, 0, stream>>>(
            d_gray, d_blur, width, height, width, width
        );
    }

    // --- Adaptive threshold (קרנל ישיר)
    {
        dim3 block(16, 16), grid((width+15)/16, (height+15)/16);
        adaptive_threshold_kernel_11_5<<<grid, block, 0, stream>>>(
            d_blur, d_thresh, width, height, width, width
        );
    }

    // --- Download thresholded image to host
    cv::Mat thresh_host(height, width, CV_8UC1);
    cudaMemcpy(thresh_host.data, d_thresh, width * height, cudaMemcpyDeviceToHost);

    // --- CPU: find contours as before
    std::vector<std::vector<cv::Point>> contours_target;
    find_contours(contours_target, thresh_host);

    // --- CPU: compare contours (CPU code לא משתנה)
    std::vector<std::vector<cv::Point>> closest_contours =
        find_closest_contours(target_frame, template_contour, contours_target, match_threshold, min_area_ratio, max_area_ratio);

    // --- free device memory
    cudaFree(d_bgr); cudaFree(d_gray); cudaFree(d_blur); cudaFree(d_thresh);

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


