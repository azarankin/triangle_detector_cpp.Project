
#pragma once
#include <shared_utils.h>
#include <debug.cuh>


__global__ void drawEquilateralTriangleKernel(uchar3* img, int width, int height, int margin);

