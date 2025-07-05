#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>


// תחליף את adaptiveThreshold על CPU בלוגיקה CUDA הזו:
void cuda_adaptive_threshold(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int blockSize, double C) {
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




__global__ void drawEquilateralTriangleKernel(uchar3* img, int width, int height, int margin)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) return;

    // --- שלב 1: רקע
    for (int x = 0; x < width; ++x) {
        img[y * width + x] = make_uchar3(255, 255, 255);
    }

    // --- שלב 2: ציור משולש רק אם בתוך ה־ROI
    int topY = margin;
    int bottomY = height - margin;
    int triHeight = bottomY - topY;

    if (y < topY || y > bottomY)
        return;

    float ratio = float(y - topY) / triHeight;
    int triWidthHalf = (width - 2 * margin) / 2;
    int centerX = width / 2;

    int x_start = int(centerX - triWidthHalf * ratio);
    int x_end   = int(centerX + triWidthHalf * ratio);

    // ציור רק בתוך תחום הצלע
    x_start = max(x_start, 0);
    x_end   = min(x_end, width - 1);

    for (int x = x_start; x <= x_end; ++x) {
        img[y * width + x] = make_uchar3(0, 0, 0);
    }
}



// __device__ bool pointInTriangle(int x, int y, int2 v0, int2 v1, int2 v2)
// {
//     float denom = (float)((v1.y - v2.y)*(v0.x - v2.x) + (v2.x - v1.x)*(v0.y - v2.y));
//     float a = ((v1.y - v2.y)*(x - v2.x) + (v2.x - v1.x)*(y - v2.y)) / denom;
//     float b = ((v2.y - v0.y)*(x - v2.x) + (v0.x - v2.x)*(y - v2.y)) / denom;
//     float c = 1.0f - a - b;
//     return a >= 0 && b >= 0 && c >= 0;
//     //float eps = 1e-3f;
//     //return a >= -eps && b >= -eps && c >= -eps;
// }

// __global__ void drawTriangleKernel(uchar3* img, int width, int height, int2 pt1, int2 pt2, int2 pt3)
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     if (x >= width || y >= height) return;

//     int idx = y * width + x;
//     uchar3 white = {255, 255, 255};
//     uchar3 black = {0, 0, 0};

//     img[idx] = white;
//     if (pointInTriangle(x, y, pt1, pt2, pt3))
//         img[idx] = black;
// }
