

#include <utils_pure_cuda.cuh>


#include <debug.cuh>




std::string utils_print_message_pure_cuda() 
{
    std::cout << "\t\t\t\t\t\t\tPrinting from CUDA OpenCV Utils library!" << std::endl;
    return "pure_cuda_utils";
}

// void cuda_gray_filter(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst)
// {
//     cv::cuda::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
//     SAVE_DEBUG_IMAGE(dst);
// }



__global__ void bgr2gray_kernel(const uchar3* src, unsigned char* dst, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    uchar3 p = src[idx];
    //unsigned char gray = (unsigned char)(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z);
    unsigned char gray = (unsigned char)((77 * (uint)p.x + 150 * (uint)p.y + 29 * (uint)p.z) >> 8);
    dst[idx] = gray;
}

void cuda_pure_gray_filter(
    const cv::Mat& src_bgr,
    cv::Mat& dst_gray,
    cudaStream_t stream
) {
    CV_Assert(src_bgr.type() == CV_8UC3);
    CV_Assert(dst_gray.type() == CV_8UC1);
    CV_Assert(src_bgr.size() == dst_gray.size());

    const int width = src_bgr.cols;
    const int height = src_bgr.rows;
    const size_t num_pixels = width * height;
    const size_t src_bytes = num_pixels * sizeof(uchar3);   // uchar3 == 3 bytes per pixel
    const size_t dst_bytes = num_pixels;

    uchar3* d_src = nullptr;
    unsigned char* d_dst = nullptr;

    cudaMalloc(&d_src, src_bytes);
    cudaMalloc(&d_dst, dst_bytes);

    // Note: reinterpret src_bgr buffer as uchar3* safely
    cudaMemcpyAsync(
        d_src,
        reinterpret_cast<const uchar3*>(src_bgr.ptr<unsigned char>()),
        src_bytes,
        cudaMemcpyHostToDevice,
        stream
    );

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    bgr2gray_kernel<<<grid, block, 0, stream>>>(d_src, d_dst, width, height);

    cudaMemcpyAsync(dst_gray.ptr<unsigned char>(), d_dst, dst_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_src);
    cudaFree(d_dst);
}






// הכרזה על d_kernel בקובץ ה־.cu (או .cuh עם extern)
__constant__ int d_kernel_int[25] = {
     1,  4,  6,  4, 1,
     4, 16, 24, 16, 4,
     6, 24, 36, 24, 6,
     4, 16, 24, 16, 4,
     1,  4,  6,  4, 1
};
// Normalization constant
#define KERNEL_SUM 256  // סכום המקדמים
#define K_ZIZE 5  // סכום המקדמים


// // פונקציה להכנה ושליחה של הגאוסיאן ל־GPU
// void upload_gaussian_kernel_5x5()
// {
//     int h_kernel[25] = {
//         1,  4,  6,  4, 1,
//         4, 16, 24, 16, 4,
//         6, 24, 36, 24, 6,
//         4, 16, 24, 16, 4,
//         1,  4,  6,  4, 1
//     };

//     // float norm = 0.0f;
//     // for (int i = 0; i < 25; ++i) norm += h_kernel[i];
//     // if (norm == 0.0f) norm = 1.0f; // הגנה מתמטית (לא צריך כאן)
//     // for (int i = 0; i < 25; ++i) h_kernel[i] /= norm;

//     // העלאה ל־constant memory במכשיר
//     cudaMemcpyToSymbol(d_kernel, h_kernel, sizeof(int) * 25);
// }


__global__ void gaussian_blur_kernel(
    const unsigned char* __restrict__ src,  // תמונת קלט בגווני אפור
    unsigned char* __restrict__ dst,        // תמונת פלט
    int width,
    int height,
    int src_stride,   // ב־bytes!
    int dst_stride    // ב־bytes!
)
{
    
    extern __shared__ unsigned char tile[]; // Allocated dynamically
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;


    const int ksize = K_ZIZE, khalf = ksize/2;
    

    // אינדקס ב־shared memory
    int tile_w = blockDim.x + 2*khalf, tile_h = blockDim.y + 2*khalf;
    int lx = threadIdx.x + khalf;
    int ly = threadIdx.y + khalf;

    // טען אזור מורחב ל־shared memory (כולל פדינג)
    // טוענים כל thread את הפיקסל שלו + גבול (ע"י לולאות, נניח)
    for (int dy = -khalf; dy <= khalf; ++dy) {
        for (int dx = -khalf; dx <= khalf; ++dx) {
            int global_x = min(max(x + dx, 0), width-1);
            int global_y = min(max(y + dy, 0), height-1);
            int shared_x = lx + dx;
            int shared_y = ly + dy;
            tile[shared_y * tile_w + shared_x] = src[global_y * src_stride + global_x];
        }
    }
    __syncthreads();



    // חישוב הקונבולוציה מתוך shared memory
    float sum = 0.0f;
    #pragma unroll
    for (int ky = -khalf; ky <= khalf; ++ky)
        #pragma unroll
        for (int kx = -khalf; kx <= khalf; ++kx)
            sum += tile[(ly+ky)*tile_w + (lx+kx)] * d_kernel_int[(ky+khalf)*ksize + (kx+khalf)];

    dst[y * dst_stride + x] = static_cast<unsigned char>((static_cast<int>(sum) + 128) >> 8);

    

    //Without shared memory, the code would look like this:
    // float sum = 0.0f;

    // #pragma unroll
    // for (int ky = -khalf; ky <= khalf; ++ky) 
    // {
    //     int yy = min(max(y + ky, 0), height - 1);
    //     #pragma unroll
    //     for (int kx = -khalf; kx <= khalf; ++kx) 
    //     {
    //         int xx = min(max(x + kx, 0), width - 1);
    //         int kval = d_kernel_int[(ky + khalf) * ksize + (kx + khalf)];
    //         sum += static_cast<int>(src[yy * src_stride + xx]) * kval;
    //     }
    // }

    // // המרה ל־uchar (עיגול)
    // //dst[y * dst_stride + x] = static_cast<unsigned char>(sum + 0.5f);
    // // נרמול לתחום 0–255
    // dst[y * dst_stride + x] = static_cast<unsigned char>((sum + (KERNEL_SUM / 2)) / KERNEL_SUM);
}




// חתימה – PURE CUDA, לא תלויית OpenCV
void pure_cuda_gaussian_blur_filter(
    const cv::Mat& src,
    cv::Mat& dst,
    cudaStream_t stream
) {
    CV_Assert(src.type() == CV_8UC1 && dst.type() == CV_8UC1);
    CV_Assert(src.size() == dst.size());

    const int width = src.cols;
    const int height = src.rows;
    const int src_stride = src.step;
    const int dst_stride = dst.step;
    const size_t src_bytes = src_stride * height;
    const size_t dst_bytes = dst_stride * height;

    unsigned char* d_src = nullptr;
    unsigned char* d_dst = nullptr;

    // Allocate GPU memory
    cudaMalloc(&d_src, src_bytes);
    cudaMalloc(&d_dst, dst_bytes);

    // Copy from host to device
    cudaMemcpyAsync(d_src, src.ptr<unsigned char>(), src_bytes, cudaMemcpyHostToDevice, stream);

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    const int ksize = K_ZIZE;  // NOTE: You must define this or pass as parameter
    const int khalf = ksize / 2;
    size_t shmem_size = (block.x + 2 * khalf) * (block.y + 2 * khalf) * sizeof(unsigned char);

    // Optional: preload kernel to constant memory
    // upload_gaussian_kernel_5x5();

    gaussian_blur_kernel<<<grid, block, shmem_size, stream>>>(
        d_src, d_dst, width, height, src_stride, dst_stride
    );

    // Copy back result
    cudaMemcpyAsync(dst.ptr<unsigned char>(), d_dst, dst_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_src);
    cudaFree(d_dst);
}





// תחליף את adaptiveThreshold על CPU בלוגיקה CUDA הזו:
// static void cuda_adaptive_threshold(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int blockSize, double C) 
// {
//     // (1) ממוצע לוקאלי – boxFilter CUDA
//     cv::cuda::GpuMat mean_local;
//     auto boxFilter = cv::cuda::createBoxFilter(src.type(), src.type(), cv::Size(blockSize, blockSize));
//     boxFilter->apply(src, mean_local);

//     // (2) מחסרים C מהמטריצה
//     cv::cuda::GpuMat threshMat;
//     cv::cuda::subtract(mean_local, cv::Scalar::all(C), threshMat);

//     // (3) משווים: src > threshMat ? 255 : 0 (THRESH_BINARY)
//     cv::cuda::compare(src, threshMat, dst, cv::CMP_GT); // dst = mask
//     // dst יוצא בינארי (0/255)
// }






__inline__ __device__ int reflect(int p, int len) {
    if (p < 0) return -p-1;
    if (p >= len) return 2*len - p - 1;
    return p;
}


#define KERNEL_SIZE 11
#define C_VALUE 5
#define BLOCK_SIZE 16

//template<int BLOCK, int KSIZE, int C>
__global__ void adaptive_threshold_kernel_11_5(
    const unsigned char* __restrict__ src, unsigned char* __restrict__ dst,
    int width, int height,
    int src_stride, int dst_stride)
{
    constexpr int KSIZE = KERNEL_SIZE;
    constexpr int KHALF = KSIZE / 2;
    // Shared memory: מותאם לדינמיות של blockDim.x/y
    __shared__ unsigned char patch[(BLOCK_SIZE + KERNEL_SIZE - 1)][(BLOCK_SIZE + KERNEL_SIZE - 1)];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int lx = threadIdx.x, ly = threadIdx.y;

    // העלאה ל־shared memory – כל thread טוען תא אחד (ולא כולם טוענים הכל!)
    #pragma unroll
    for (int dy = 0; dy < KSIZE; ++dy) {
        #pragma unroll
        for (int dx = 0; dx < KSIZE; ++dx) {
            int gx = x - KHALF + dx;
            int gy = y - KHALF + dy;
            int sx = lx + dx, sy = ly + dy;
            if (sx < blockDim.x + KSIZE - 1 && sy < blockDim.y + KSIZE - 1) 
            {
                int imgx = reflect(gx, width);
                int imgy = reflect(gy, height);
                patch[sy][sx] = src[imgy * src_stride + imgx];
            }
        }
    }
    __syncthreads();

    // טיפול בפיקסל ליבה
    if (x < width && y < height && lx < blockDim.x && ly < blockDim.y) 
    {
        float sum = 0.0f;
        #pragma unroll
        for (int ky = 0; ky < KSIZE; ++ky)
            #pragma unroll
            for (int kx = 0; kx < KSIZE; ++kx)
                sum += patch[ly + ky][lx + kx];

        float mean = sum / (KSIZE * KSIZE);
        unsigned char srcval = patch[ly + KHALF][lx + KHALF];
        dst[y * dst_stride + x] = (srcval > mean - C_VALUE) ? 255 : 0;
    }
}





void pure_cuda_adaptive_threshold(
    const cv::Mat& src,
    cv::Mat& dst,
    cudaStream_t stream
) {
    CV_Assert(src.type() == CV_8UC1 && dst.type() == CV_8UC1);
    CV_Assert(src.rows == dst.rows && src.cols == dst.cols);

    const int width = src.cols;
    const int height = src.rows;
    const int src_stride = src.step;
    const int dst_stride = dst.step;
    const size_t src_bytes = src_stride * height;
    const size_t dst_bytes = dst_stride * height;

    unsigned char* d_src = nullptr;
    unsigned char* d_dst = nullptr;

    cudaMalloc(&d_src, src_bytes);
    cudaMalloc(&d_dst, dst_bytes);

    cudaMemcpyAsync(d_src, src.ptr<unsigned char>(), src_bytes, cudaMemcpyHostToDevice, stream);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    adaptive_threshold_kernel_11_5<<<grid, block, 0, stream>>>(
        d_src, d_dst, width, height, src_stride, dst_stride
    );

    cudaMemcpyAsync(dst.ptr<unsigned char>(), d_dst, dst_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_src);
    cudaFree(d_dst);
}










__global__ void threshold_kernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int width, int height,
    int src_stride, int dst_stride,
    unsigned char thresh, unsigned char maxval
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    unsigned char val = src[y * src_stride + x];
    dst[y * dst_stride + x] = (val >= thresh) ? maxval : 0;
}



void pure_cuda_threshold_filter(
    const cv::Mat& src,
    cv::Mat& dst,
    cudaStream_t stream,
    unsigned char thresh,
    unsigned char maxval
) {
    CV_Assert(src.type() == CV_8UC1 && dst.type() == CV_8UC1);
    CV_Assert(src.rows == dst.rows && src.cols == dst.cols);

    const int width = src.cols;
    const int height = src.rows;
    const int src_stride = src.step;
    const int dst_stride = dst.step;
    const size_t src_bytes = src_stride * height;
    const size_t dst_bytes = dst_stride * height;

    unsigned char* d_src = nullptr;
    unsigned char* d_dst = nullptr;

    cudaMalloc(&d_src, src_bytes);
    cudaMalloc(&d_dst, dst_bytes);

    cudaMemcpyAsync(d_src, src.ptr<unsigned char>(), src_bytes, cudaMemcpyHostToDevice, stream);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1)/block.x, (height + block.y - 1)/block.y);

    threshold_kernel<<<grid, block, 0, stream>>>(
        d_src, d_dst, width, height, src_stride, dst_stride, thresh, maxval
    );

    cudaMemcpyAsync(dst.ptr<unsigned char>(), d_dst, dst_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_src);
    cudaFree(d_dst);
}
