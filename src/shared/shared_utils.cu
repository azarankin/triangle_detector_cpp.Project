
#include <shared_utils.cuh>

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


