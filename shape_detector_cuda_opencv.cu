#include "shape_detector_cuda_opencv.cuh"
#include "utils/utils_cuda_opencv.cuh"


#include <cuda_runtime.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp> // cv::cuda::cvtColor
#include <opencv2/cudafilters.hpp>



DebugConfig g_debug_config;


void save_debug_image(const std::string& name, const cv::Mat& img)
{

if(!g_debug_config.enable)
    return;
if (img.empty()) return;
//td::filesystem::create_directories(out_dir);
cv::Mat to_save;
if (img.channels() == 4)
    cv::cvtColor(img, to_save, cv::COLOR_BGRA2RGBA);
else if (img.channels() == 3)
    cv::cvtColor(img, to_save, cv::COLOR_BGR2RGB);
else
    to_save = img;

cv::imwrite(g_debug_config.output_dir + (!g_debug_config.prefix.empty() ? ("/" + g_debug_config.prefix +  "__") : "/") + name + ".png", to_save);

}

void save_debug_image(const std::string& name, const cv::cuda::GpuMat& gpu_img)
{
if(!g_debug_config.enable)
    return;
cv::Mat cpu_img;
gpu_img.download(cpu_img);
save_debug_image(name, cpu_img);
}





std::string print_message() 
{
    std::cout << "\t\t\t\t\t\t\tPrinting from CUDA library!" << std::endl;
    return "cuda_opencv";
}


void imshow(const std::string& title, const cv::Mat& img) 
{
    cv::Mat to_show;
    if (img.channels() == 4) {
        cv::cvtColor(img, to_show, cv::COLOR_BGRA2RGBA);
    } else if (img.channels() == 3) {
        cv::cvtColor(img, to_show, cv::COLOR_BGR2RGB);
    } else {
        to_show = img.clone();
    }
    cv::imshow(title, to_show);
    cv::waitKey(0);  // waiting for key pressing
}






cv::Mat draw_contour_on_image(cv::Mat& img, const std::vector<cv::Point>& contour, const cv::Scalar& color, int thickness)
{
    if (img.empty())
        throw std::runtime_error("Failed to load image");

    std::vector<std::vector<cv::Point>> contours{contour};
    cv::drawContours(img, contours, 0, color, thickness);
    return img;
}

cv::Mat draw_contour_on_image(cv::Mat& img, const std::vector<std::vector<cv::Point>>& contours, const cv::Scalar& color, int thickness) 
{
    if (img.empty())
        throw std::runtime_error("Failed to load image");
    cv::drawContours(img, contours, -1, color, thickness);
    return img;
}









std::vector<cv::Point> find_shape_contour(const std::string& address)
{
    // Load image
    cv::Mat host_img = cv::imread(address);
    if (host_img.empty())
        throw std::runtime_error("Failed to load image: " + address);

    // Upload to GPU
    cv::cuda::GpuMat gpu_img(host_img);

    // Convert to grayscale
    cv::cuda::GpuMat gpu_gray;
    cv::cuda::cvtColor(gpu_img, gpu_gray, cv::COLOR_BGR2GRAY);
    save_debug_image("find_shape_contour__gray", gpu_gray);

    // Gaussian blur
    cv::cuda::GpuMat gpu_blur;
    auto gauss_filter = cv::cuda::createGaussianFilter(gpu_gray.type(), gpu_gray.type(), cv::Size(5, 5), 0);
    gauss_filter->apply(gpu_gray, gpu_blur);
    save_debug_image("find_shape_contour__blur", gpu_blur);

    // Threshold
    cv::cuda::GpuMat gpu_thresh;
    cv::Mat thresh1;
    cv::cuda::threshold(gpu_blur, gpu_thresh, 127, 255, cv::THRESH_BINARY);
    gpu_thresh.download(thresh1);
    save_debug_image("find_shape_contour__thresh", thresh1);

    // Find contours (CPU only)
    std::vector<std::vector<cv::Point>> contours_template;
    cv::findContours(thresh1, contours_template, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // Sort contours by area (descending)
    std::sort(contours_template.begin(), contours_template.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return cv::contourArea(a) > cv::contourArea(b);
        });

    if (contours_template.size() < 2)
        throw std::runtime_error("Template image must contain at least two contours.");

    return contours_template[1];
}









std::vector<std::vector<cv::Point>> contour_compare(
    cv::Mat& target_frame, 
    const std::vector<cv::Point>& template_contour,
    double match_threshold, 
    double min_area_ratio, 
    double max_area_ratio)
{
    // Upload image to GPU
    cv::cuda::GpuMat gpu_frame(target_frame);

    // Convert to grayscale
    cv::cuda::GpuMat gpu_gray;
    cv::cuda::cvtColor(gpu_frame, gpu_gray, cv::COLOR_BGR2GRAY);
    save_debug_image("contour_compare__gray", gpu_gray);


    // Gaussian blur (CUDA)
    cv::cuda::GpuMat gpu_blur;
    auto gauss_filter = cv::cuda::createGaussianFilter(gpu_gray.type(), gpu_gray.type(), cv::Size(5, 5), 0);
    gauss_filter->apply(gpu_gray, gpu_blur);
    save_debug_image("contour_compare__blur", gpu_blur);


    // --------- *** NEW: CUDA adaptive threshold *** ---------
    cv::cuda::GpuMat gpu_thresh;
    cuda_adaptive_threshold(gpu_blur, gpu_thresh, /*blockSize=*/11, /*C=*/5);
    save_debug_image("contour_compare__thresh", gpu_thresh);


    // הורדה ל-CPU להמשך עיבוד (contours)
    cv::Mat thresh2;
    gpu_thresh.download(thresh2);

    // Contours (CPU)
    std::vector<std::vector<cv::Point>> contours_target;
    cv::findContours(thresh2, contours_target, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    double frame_area = static_cast<double>(target_frame.rows * target_frame.cols);
    double min_area = frame_area * min_area_ratio;
    double max_area = frame_area * max_area_ratio;

    std::vector<std::vector<cv::Point>> closest_contours;
    for (const auto& c : contours_target) {
        double match = cv::matchShapes(template_contour, c, cv::CONTOURS_MATCH_I3, 0.0);
        if (match < match_threshold) {
            double area = cv::contourArea(c);
            if (area < min_area || area > max_area)
                continue;
            closest_contours.push_back(c);
        }
    }
    return closest_contours;
}







void run_camera(const std::vector<cv::Point>& template_contour, int camid) 
{
    cv::VideoCapture cap(camid);

    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot access the camera." << std::endl;
        return;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Failed to grab frame" << std::endl;
            break;
        }

        // קריאה לפונקציית ההשוואה
        auto contours = contour_compare(frame, template_contour);
        auto frame_with_contours = draw_contour_on_image(frame, contours);
        // (הפונקציה אמורה לצייר את הקונטורים על frame עצמו)

        cv::imshow("Live Feed", frame_with_contours);

        char key = (char)cv::waitKey(1);
        if (key == 27 || key == 'q') { // Esc או q
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
}






void create_triangle_image(const std::string& filename, cv::Size size, int margin) 
{
    int width = size.width;
    int height = size.height;

    uchar3* d_img;
    size_t imgSize = width * height * sizeof(uchar3);
    cudaMalloc(&d_img, imgSize);

    int2 pt1 = make_int2(width / 2, margin);                  // top
    int2 pt2 = make_int2(margin, height - margin);            // left
    int2 pt3 = make_int2(width - margin, height - margin);    // right


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