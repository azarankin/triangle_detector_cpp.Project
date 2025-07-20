// borders_cuda.cuh

#pragma once
#include <cuda_runtime.h>
#include <cassert>
#include <math.h>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//using namespace cv;
// vector<vector<Point> > contours;
// vector<Vec4i> hierarchy;




typedef unsigned char byte;
#define M 600//1232              //dimensions of image
#define N 600//1028
#define Mg 648//1248             //  first multiple of N_BLOCKS_ROWS greater than 1232 (M)
#define Ng 648//1056             //  first multiple of N_BLOCKS_COLS greater than 1028 (N) 
#define N_BLOCKS_ROWS 16	  // 4
#define N_BLOCKS_COLS 16    // 4
#define MAX_N_BORDS 500     //Max number of borders by rectangle of image, in this version
// Maximum number of threads per block of K20Xm: 1024 = 32*32
#define THREADS_PER_BLOCK_X 32    //Para preproceso
#define THREADS_PER_BLOCK_Y 32    //Para preproceso


struct coord { int i, j; };
struct VecCont { coord act, sig, ant; int next; };
// structure to store coordinate of each point in a contour
 // coord act - Coordenadas del punto actual
 // coord sig - Coordenadas del punto siguiente en, closed, or covered)


typedef enum { OPEN_CONTOUR, CLOSED_CONTOUR, COVERED_CONTOUR } con;


struct IndCont{ int ini; int fin; con sts; };
// structure of positions of ech contour
   // int ini - Index of initial point of in a vector of veccont
   // int fin - Index of final point of in a vector of veccont
   // contour state (open, closed, or covered)


// הצהרות לקרנלים (צריך לכלול או להעתיק מהקובץ הקיים)
__global__ void preprocessing_gpu(byte*, byte*);
__global__ void parallel_tracking(byte*, byte*, int*, VecCont*, IndCont*, IndCont*, int*);
__global__ void vertical_connection(int, int*, VecCont*, IndCont*, int*, IndCont*, int*, int, int, IndCont*, int*);
__global__ void horizontal_connection(int, int*, VecCont*, IndCont*, int*, IndCont*, int*, int, int, IndCont*, int*);
__global__ void plot_contours_gpu(IndCont*, VecCont*, byte*);
__global__ void borde_ceros(byte*);

void copy_p_a_g(byte* h_A, byte* h_Ag);


// פונקציה עוטפת מודרנית:
void find_contours_cuda(std::vector<std::vector<cv::Point>>& contours, const cv::Mat& binary);

