
#pragma once
#include <shared_utils.h>
#include <debug.cuh>
//#include <cuda_runtime.h>

__global__ void drawEquilateralTriangleKernel(uchar3* img, int width, int height, int margin);

