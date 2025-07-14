
/**
 * Parallel CUDA algorithm for border tracking Víctor Manuel García Mollá (vmgarcia@dsic.upv.es)
 */

#include <borders_cuda.cuh>
#include <iostream>



// פונקציה עוטפת מודרנית:
void find_contours_cuda(std::vector<std::vector<cv::Point>>& contours, const cv::Mat& binary)
{
    // --- הכנה --- //
    assert(binary.type() == CV_8UC1);
    int mat_mem_size = sizeof(byte) * M * N;
    int mat_mem_sizeg = sizeof(byte) * Mg * Ng;
    int blq_mem_size  = sizeof(int) * N_BLOCKS_ROWS * N_BLOCKS_COLS;
    int vec_mem_size = sizeof(VecCont) * Mg * Ng * 2;
    int ind_mem_size  = sizeof(IndCont)  * N_BLOCKS_ROWS * N_BLOCKS_COLS * MAX_N_BORDS;
    int marked_mem_size  = sizeof(int)  * N_BLOCKS_ROWS * N_BLOCKS_COLS * MAX_N_BORDS;
    int ind_mem_size_glob  = sizeof(IndCont)  * 100000;

    // --- הקצאת זיכרון --- //
    byte *h_A  = (byte*)malloc(mat_mem_size);
    byte *h_Ag = (byte*)malloc(mat_mem_sizeg);

    // קלט – תמונה ממוזערת לפורמט של האלגוריתם
    // (לפי M, N)
    cv::Mat srcROI = binary(cv::Rect(0,0,M,N));
    memcpy(h_A, srcROI.data, mat_mem_size);
    memset(h_Ag, 0, mat_mem_sizeg);
    copy_p_a_g(h_A, h_Ag);

    // Output buffers (CPU)
    int *h_numconts_glob = (int*)malloc(sizeof(int));
    VecCont *h_vec_conts = (VecCont*)malloc(vec_mem_size);
    IndCont *h_ind_conts_glob = (IndCont*)malloc(ind_mem_size_glob);

    // --- הקצאת GPU --- //
    byte *d_A, *d_is_bord;
    int *d_numconts, *d_numconts_aux, *d_numconts_glob;
    VecCont *d_vec_conts;
    IndCont *d_ind_conts, *d_ind_conts_glob, *d_ind_conts_aux;

    cudaMalloc((void**)&d_A, mat_mem_sizeg);
    cudaMalloc((void**)&d_is_bord, mat_mem_sizeg);
    cudaMalloc((void**)&d_numconts, blq_mem_size);
    cudaMalloc((void**)&d_numconts_aux, blq_mem_size);
    cudaMalloc((void**)&d_numconts_glob, sizeof(int));
    cudaMalloc((void**)&d_vec_conts, vec_mem_size);
    cudaMalloc((void**)&d_ind_conts, ind_mem_size);
    cudaMalloc((void**)&d_ind_conts_aux, ind_mem_size);
    cudaMalloc((void**)&d_ind_conts_glob, ind_mem_size_glob);

    // --- העתקת נתונים ל-GPU --- //
    cudaMemcpy(d_A, h_Ag, mat_mem_sizeg, cudaMemcpyHostToDevice);
    cudaMemset(d_is_bord, 0, mat_mem_sizeg);
    cudaMemset(d_numconts, 0, blq_mem_size);
    cudaMemset(d_numconts_aux, 0, blq_mem_size);
    cudaMemset(d_numconts_glob, 0, sizeof(int));
    cudaMemset(d_vec_conts, 0, vec_mem_size);
    cudaMemset(d_ind_conts, 0, ind_mem_size);
    cudaMemset(d_ind_conts_aux, 0, ind_mem_size);
    cudaMemset(d_ind_conts_glob, 0, ind_mem_size_glob);

    // --- קרנלים --- //
    // שמירה על מבנה כמו במימוש המקורי
    borde_ceros<<<1, 1024>>>(d_A);

    dim3 dimBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 dimGrid((M/dimBlock.x)+1, (N/dimBlock.y)+1);
    preprocessing_gpu<<<dimGrid, dimBlock>>>(d_A, d_is_bord);

    dimGrid.x = N_BLOCKS_ROWS;
    dimGrid.y = N_BLOCKS_COLS;
    dimBlock.x = 1;
    dimBlock.y = 1;
    parallel_tracking<<<dimGrid, dimBlock>>>(d_A, d_is_bord, d_numconts, d_vec_conts, d_ind_conts, d_ind_conts_glob, d_numconts_glob);

    // חיבור קונטורים בין בלוקים – בדיוק לפי המקור
    int mbn = dimGrid.x/2;
    int num_max_c_etapa = MAX_N_BORDS;
    int numfbl = (Mg/N_BLOCKS_ROWS);
    int numcbl = (Ng/N_BLOCKS_COLS);
    while(mbn >= 1) {
        dimGrid.x = mbn;
        vertical_connection<<<dimGrid, dimBlock>>>(num_max_c_etapa, d_numconts, d_vec_conts, d_ind_conts, d_numconts_aux, d_ind_conts_aux, nullptr, numfbl, numcbl, d_ind_conts_glob, d_numconts_glob);
        cudaDeviceSynchronize();
        std::swap(d_numconts, d_numconts_aux);
        std::swap(d_ind_conts, d_ind_conts_aux);
        num_max_c_etapa *= 2;
        numfbl *= 2;
        mbn /= 2;
    }
    dimGrid.x = 1;
    int nbn = dimGrid.y/2;
    while(nbn >= 1) {
        dimGrid.y = nbn;
        horizontal_connection<<<dimGrid, dimBlock>>>(num_max_c_etapa, d_numconts, d_vec_conts, d_ind_conts, d_numconts_aux, d_ind_conts_aux, nullptr, numfbl, numcbl, d_ind_conts_glob, d_numconts_glob);
        cudaDeviceSynchronize();
        std::swap(d_numconts, d_numconts_aux);
        std::swap(d_ind_conts, d_ind_conts_aux);
        num_max_c_etapa *= 2;
        numcbl *= 2;
        nbn /= 2;
    }

    // --- העתקת פלט ל-host --- //
    cudaMemcpy(h_numconts_glob, d_numconts_glob, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ind_conts_glob, d_ind_conts_glob, ind_mem_size_glob, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vec_conts, d_vec_conts, vec_mem_size, cudaMemcpyDeviceToHost);

    // --- פיענוח תוצאה ל-vector<vector<cv::Point>> --- //
    contours.clear();
    int num_contours = h_numconts_glob[0];
    const int MAX_IDX = 10 /* כאן תכניס את גודל המערך שלך */;
    for(int c=1; c <= num_contours; ++c) {
        const IndCont& cont = h_ind_conts_glob[c];
        std::vector<cv::Point> points;
        int idx = cont.ini;
        int steps = 0;
        while (true) {
            if (idx < 0 || idx >= MAX_IDX) {
                std::cerr << "Error: idx " << idx << " out of bounds in contour " << c << std::endl;
                break;
            }
            points.push_back({h_vec_conts[idx].act.j, h_vec_conts[idx].act.i});
            if (idx == cont.fin)
                break;
            idx = h_vec_conts[idx].next;
            if (++steps > 10000) { // להימנע מלולאה אינסופית
                std::cerr << "Error: possible infinite loop in contour " << c << std::endl;
                break;
            }
        }
        contours.push_back(points);
    }

    // --- שחרור זיכרון --- //
    cudaFree(d_A); cudaFree(d_is_bord); cudaFree(d_numconts); cudaFree(d_numconts_aux); cudaFree(d_numconts_glob);
    cudaFree(d_vec_conts); cudaFree(d_ind_conts); cudaFree(d_ind_conts_aux); cudaFree(d_ind_conts_glob);
    free(h_A); free(h_Ag); free(h_numconts_glob); free(h_vec_conts); free(h_ind_conts_glob);
}



#define CUDA_SAFE_CALL( call, routine ) { \
 cudaError_t err = call; \
 if( cudaSuccess != err ) { \
   fprintf(stderr,"CUDA: error %d occurred in %s routine. Exiting...\n", err, routine); \
   exit(err); \
 } \


// data of test image
//d_A is the original image; and h_A is the original size output image,
#define	h_A(i,j)		h_A[ (i) + ((j)*(M)) ]
#define	h_Ag(i,j)		h_Ag[ (i) + ((j)*(Mg)) ]
#define	h_Asal(i,j) 		h_Asal[ (i) + ((j)*(M)) ]
#define	d_A(i,j) 		d_A[ (i) + ((j)*(Mg)) ]  //#define	d_A(i,j) 		d_A[ (i) + ((j)*(M)) ]
#define	d_Ag(i,j,ldg) 		d_Ag[ (i) + ((j)*(Mg)) ]
#define	d_Asal(i,j) 		d_Asal[ (i) + ((j)*(M)) ]
#define	d_is_bord(i,j) 		d_is_bord[ (i) + ((j)*(Mg)) ]

#define indice_despl_x(i,j)	indice_despl_x[ (i) + ((j)*(8)) ]


/**
 * Kernel CUDA: detection of contour points
 */
__global__ void preprocessing_gpu(
	byte *d_A,   //input image
	byte *d_is_bord  //output binary array, indicating whether pixel (i,j) is contour (d_is_bord(i,j)=1) or not (d_is_bord(i,j)=0)
) {
	__shared__ byte arr_sh[(THREADS_PER_BLOCK_X + 2)*(THREADS_PER_BLOCK_Y + 2)];
//	buffer of shared memory
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int pos = i + 1 + Mg * (j + 1);
	int pos_local = threadIdx.x + 1 + (threadIdx.y + 1)*(blockDim.x + 2);
	int ilocal = threadIdx.x + 1;
	int jlocal = threadIdx.y + 1;
	int lda_shared = blockDim.x + 2;
	int cond, pos_zero = 0;

    //load piece of image on shared memory buffer
	if ((i < M - 1) && (j < N - 1))
	{
		arr_sh[pos_local] = d_A[pos];
		if (threadIdx.x == 0)
			arr_sh[ilocal - 1 + (jlocal)*lda_shared] = d_A[i + Mg * (j + 1)];
		if (threadIdx.x == blockDim.x - 1)
			arr_sh[ilocal + 1 + (jlocal)*lda_shared] = d_A[i + 2 + Mg * (j + 1)];
		if (threadIdx.y == 0)
			arr_sh[ilocal + (jlocal - 1)*(lda_shared)] = d_A[i + 1 + Mg * (j)];
		if (threadIdx.y == blockDim.y - 1)
			arr_sh[ilocal + (jlocal + 1)*(lda_shared)] = d_A[i + 1 + Mg * (j + 2)];
		if ((threadIdx.x == 0) && (threadIdx.y == 0))
			arr_sh[ilocal - 1 + (jlocal - 1)*(lda_shared)] = d_A[i + Mg * (j)];
		if ((threadIdx.x == 0) && (threadIdx.y == blockDim.y - 1))
			arr_sh[ilocal - 1 + (jlocal + 1)*(lda_shared)] = d_A[i + Mg * (j + 2)];
		if ((threadIdx.x == blockDim.x - 1) && (threadIdx.y == blockDim.y - 1))
			arr_sh[ilocal + 1 + (jlocal + 1)*(lda_shared)] = d_A[i + 2 + Mg * (j + 2)];
		if ((threadIdx.x == blockDim.x - 1) && (threadIdx.y == 0))
			arr_sh[ilocal + 1 + (jlocal - 1)*(lda_shared)] = d_A[i + 2 + Mg * (j)];
	
	}
  __syncthreads();
	
	if ((i < M - 1) && (j < N - 1))
   //determine whether pixel(i,j) is a contour pixel
	{
		pos_zero = (arr_sh[ilocal - 1 + (jlocal)*(lda_shared)] == 0) * 2;
		cond = (pos_zero == 0)*(arr_sh[ilocal + (jlocal + 1)*(lda_shared)] == 0);
		pos_zero = 4 * cond + (1 - cond)*pos_zero;
		cond = (pos_zero == 0)*(arr_sh[ilocal + 1 + (jlocal)*(lda_shared)] == 0);
		pos_zero = 6 * cond + (1 - cond)*pos_zero;
		cond = (pos_zero == 0)*(arr_sh[ilocal + (jlocal - 1)*(lda_shared)] == 0);
		pos_zero = 8 * cond + (1 - cond)*pos_zero;
		d_is_bord[pos] = (arr_sh[pos_local] > 0) && (pos_zero > 0);
	

	}
	
}


__device__ void  clockwise_2(int *difi, int *difj, int *iout, int *jout, int* pos)
// function for obtaining next pixel rotating clockwise
 //difi, dif j give relative position of actual pixel relative to center pixel
 // example, if center pixel is (i,j) and difi=1, dif j=0 , present pixel is              
//i+1, j+0.
//pos =2 =>(0, -1); pos=3 =>(-1, 0), pos =5 => 0,1, pos=7 => 1,0
{
 if (*difi==1)
   {  if (*difj==1)
          {*iout=1;
          *jout=0;
          *pos=7;
		  }
     else if(*difj==0)
          {*iout=1;
          *jout=-1;
          *pos=1;
		  }
     else
           {*iout=0;
           *jout=-1;
             *pos=2;
		 }
    }
 else if(*difi==0)
     {
       if (*difj==-1)
         {  *iout=-1;
          *jout=-1;
            *pos=1;
	      }		
     else if(*difj==1) 
	      {
          *iout=1;
          *jout=1;
            *pos=1;
		  }
     }
 else if (*difi==-1)
     {if (*difj==-1)
         {*iout=-1;
         *jout=0;
           *pos=3;
		  } 
     else if(*difj==0)
         {*iout=-1;
         *jout=1;
           *pos=1;
		  } 
     else if (*difj==1)    
         {*iout=0;
         *jout=1;
         *pos=5;
		 }
     }
  
}
__device__ void  counterclock_2(int *difi, int *difj, int *iout, int *jout, int* pos)
// function for obtaining next pixel rotating counterclockwise
 //difi, dif j give relative position of actual pixel relative to center pixel
 // example, if center pixel is (i,j) and difi=1, dif j=0 , present pixel is              
//i+1, j+0.
//pos =2 =>0, -1; pos=3 =>-1, 0, pos =5 => 0,1, pos=7 => 1,0
 {if (*difi==1)
     {if (*difj==1)
       {   *iout=0;
          *jout=1;
          *pos=5; 
		}  
     else if(*difj==0)
         { *iout=1;
          *jout=1;
          *pos=1;
		  }
     else
	     {
           *iout=1;
           *jout=0;
           *pos=7;
		   }
     }
 else if(*difi==0)
    { if (*difj==-1)
         { *iout=1;
          *jout=-1;
          *pos=1;
		  }
     else if(*difj==1) 
          {*iout=-1;
          *jout=1;
          *pos=1;
		  }
     }
 else if (*difi==-1)
     {if (*difj==-1)
         {*iout=0;
         *jout=-1;
         *pos=2;
		} 
     else if(*difj==0)
         {*iout=-1;
         *jout=-1;
         *pos=1;
		 }
     else if (*difj==1)    
         {*iout=-1;
         *jout=0;
         *pos=3;
		 }
     
     } 
}


__device__ void track_fw_bkw(
	int *i_vec_conts_ini, //initial point of d_vec_conts vector where present contour is stored
	byte *d_A,        //image
	byte *d_is_bord,  //is_bord array
	int* d_numconts,  //index of contour being tracked
	VecCont* d_vec_conts, //vector of contour points
	IndCont* d_ind_conts,  //vector of contours
	int i_ind_conts,  //number of contour, already increased
	int i_ini,       //boundaries of rectangle i_ini,i_fin, j_ini, j_fin
	int j_ini,
	int i_fin,
	int j_fin,
	coord c_ini_ant,   //triad for start of tracking
	coord c_ini_act,
	coord c_ini_sig
)
{
	coord coord_sig, coord_act, coord_ant;
	int dif_i, dif_j, itcount, i_vec_conts, iaux, val;
	int	found, jaux, pos;
	i_vec_conts = *i_vec_conts_ini;

	coord_sig = c_ini_sig;
	coord_act = c_ini_act;
	//set first point of border
	d_vec_conts[i_vec_conts].act = c_ini_act;
	d_vec_conts[i_vec_conts].ant = c_ini_ant;
	d_vec_conts[i_vec_conts].sig = c_ini_sig;
	d_ind_conts[i_ind_conts].ini = i_vec_conts;
	d_ind_conts[i_ind_conts].fin = i_vec_conts;
	d_ind_conts[i_ind_conts].sts = OPEN_CONTOUR;
	
	int end_track_forward = 0;
	//start tracking forward, checking if we are leaving the rectangle 
	if ((coord_sig.i < i_ini) || (coord_sig.i > i_fin) || (coord_sig.j < j_ini) || (coord_sig.j > j_fin))
	{
		d_vec_conts[i_vec_conts].next = 0; //leaving the rectangle

		end_track_forward = 2;
	}
	else
	{
		while (end_track_forward == 0)
		{  //if we do not leave, add new point to contour
			i_vec_conts++;
			d_vec_conts[i_vec_conts].act = coord_sig;
			d_vec_conts[i_vec_conts].ant = coord_act;
			d_vec_conts[i_vec_conts - 1].next = i_vec_conts;
			coord_ant = coord_act;
			coord_act = coord_sig;
		

			dif_i = coord_ant.i - coord_act.i;
			dif_j = coord_ant.j - coord_act.j;
			found = 0;
			itcount = 0;
			val = d_A(coord_act.i, coord_act.j);
			//next while look for next "one" after (iant,jant), 
			//and stores in  "val" the zeros found,  for updating pixel value, as in algorithm 3 in paper
			while ((found == 0) && (itcount <= 8))
			{
				counterclock_2(&dif_i, &dif_j, &iaux, &jaux, &pos);
				if (d_A(coord_act.i + iaux, coord_act.j + jaux) == 0)
				{
					val = val * pos;
					dif_i = iaux;
					dif_j = jaux;
					itcount++;
				}
				else
				{
					found = 1;
					coord_sig.i = coord_act.i + iaux;
					coord_sig.j = coord_act.j + jaux;
				}
			}
			if ((coord_sig.i < i_ini) || (coord_sig.i > i_fin) || (coord_sig.j < j_ini) || (coord_sig.j > j_fin))
			{  //if leaving rectangle, finish this piece of contour
				end_track_forward = 2;
				d_vec_conts[i_vec_conts].sig = coord_sig;
				d_ind_conts[i_ind_conts].fin = i_vec_conts;          
				d_A(coord_act.i, coord_act.j) = val;
			}

			else
			{
				d_vec_conts[i_vec_conts].sig = coord_sig;
			
				d_A(coord_act.i, coord_act.j) = val;
			}
			//If contour "closes":
			if ((coord_sig.i == c_ini_act.i) && (coord_sig.j == c_ini_act.j) && (coord_act.i == c_ini_ant.i) && (coord_act.j == c_ini_ant.j))
			{
				end_track_forward = 1;

				d_ind_conts[i_ind_conts].sts = CLOSED_CONTOUR;
				d_ind_conts[i_ind_conts].fin = i_vec_conts;
				d_vec_conts[i_vec_conts].next = (*i_vec_conts_ini);
				d_vec_conts[i_vec_conts].sig = c_ini_act;
				d_vec_conts[(*i_vec_conts_ini)].ant = coord_act;
			}
		}
	}
	if (end_track_forward == 2)
	{ //the contour went out of rectangle; we go back to beginning and track backwards; very similar, but backwards
		coord_ant = c_ini_ant;
		coord_act = c_ini_act;
		coord_sig = c_ini_sig;
		int anterior = d_ind_conts[i_ind_conts].ini;
		if ((coord_ant.i < i_ini) || (coord_ant.i > i_fin) || (coord_ant.j < j_ini) || (coord_ant.j > j_fin))

			d_vec_conts[(*i_vec_conts_ini)].ant = coord_ant;
		else
		{
			int end_track_backward = 0;
			while (end_track_backward == 0)
			{

				i_vec_conts++;
				d_vec_conts[i_vec_conts].act = coord_ant;
				d_vec_conts[i_vec_conts].sig = coord_act;
				d_vec_conts[i_vec_conts].next = anterior;
				anterior = i_vec_conts;
				coord_sig = coord_act;
				coord_act = coord_ant;
				dif_i = coord_sig.i - coord_act.i;
				dif_j = coord_sig.j - coord_act.j;
				found = 0;
				itcount = 0;
				val = d_A(coord_act.i, coord_act.j);
				while (found == 0 && itcount <= 8) //rotating, looking for next "backward" pixel
				{
					clockwise_2(&dif_i, &dif_j, &iaux, &jaux, &pos);
					if (d_A(coord_act.i + iaux, coord_act.j + jaux) == 0)
					{
						val = val * pos;
						dif_i = iaux;
						dif_j = jaux;
						itcount++;
					}
					else
					{
						found = 1;
						coord_ant.i = coord_act.i + iaux;
						coord_ant.j = coord_act.j + jaux;
					}
				}
				if ((coord_ant.i < i_ini) || (coord_ant.i > i_fin) || (coord_ant.j < j_ini) || (coord_ant.j > j_fin))
				{
					end_track_backward = 2;
					d_vec_conts[i_vec_conts].ant = coord_ant;
					d_ind_conts[i_ind_conts].ini = i_vec_conts;
					d_A(coord_act.i, coord_act.j) = val;
				}
				else
				{
					d_vec_conts[i_vec_conts].ant = coord_ant;
					d_ind_conts[i_ind_conts].ini = i_vec_conts;
					d_A(coord_act.i, coord_act.j) = val;
				}

			}
		}

	}
	(*i_vec_conts_ini) = i_vec_conts;
}
/**auxiliar kernels for parallel tracking */

__device__ void  clockwise_o(int *difi, int *difj, int *iout, int *jout) 
{
	if (*difi == 1)
	{
		if (*difj == 1)
		{
			*iout = 1;
			*jout = 0;

		}
		else if (*difj == 0)
		{
			*iout = 1;
			*jout = -1;

		}
		else
		{
			*iout = 0;
			*jout = -1;

		}
	}
	else if (*difi == 0)
	{
		if (*difj == -1)
		{
			*iout = -1;
			*jout = -1;

		}
		else if (*difj == 1)
		{
			*iout = 1;
			*jout = 1;

		}
	}
	else if (*difi == -1)
	{
		if (*difj == -1)
		{
			*iout = -1;
			*jout = 0;

		}
		else if (*difj == 0)
		{
			*iout = -1;
			*jout = 1;

		}
		else if (*difj == 1)
		{
			*iout = 0;
			*jout = 1;

		}
	}

}
__device__ void rotate_ini(byte *d_A, coord *coord_ant, coord *coord_sig, int *found, int *val, int *pos_ult_cero, coord coord_act)
{
    //first rotation around a pixel, looking for a first not covered triad to start tracking it;
    //line 5 , Algorithm 1 of paper
	int dif_i = 0, dif_j = -1;
	int itcount, iaux, jaux, iaux2, jaux2;
	if (d_A(coord_act.i, coord_act.j - 1) != 0)
	{
		*found = 0;
		itcount = 0;
		while (*found == 0 && itcount <= 4)
		{
			clockwise_o(&dif_i, &dif_j, &iaux2, &jaux2);
			clockwise_o(&iaux2, &jaux2, &iaux, &jaux);
			if (d_A(coord_act.i + iaux, coord_act.j + jaux) == 0)
			{
				*found = 1;
				dif_i = iaux;
				dif_j = jaux;
			}
			else
			{
				dif_i = iaux;
				dif_j = jaux;
				itcount = itcount + 1;
			}
		}
	}
	*found = 0;
	itcount = 0;
	while (*found == 0 && itcount <= 8) //lookfor pixel greater than zero, clockwise
	{
		clockwise_o(&dif_i, &dif_j, &iaux, &jaux);
		if (d_A(coord_act.i + iaux, coord_act.j + jaux) != 0)
		{
			*found = 1;
			(*coord_ant).i = coord_act.i + iaux;
			(*coord_ant).j = coord_act.j + jaux;
		}
		else
		{
			dif_i = iaux;
			dif_j = jaux;
			itcount = itcount + 1;
		}
	}
	if (*found == 0) //NO neighbor pixel greater than zero
	{
		(*coord_ant).i = 0;
		(*coord_ant).j = 0;
		(*coord_sig).i = 0;
		(*coord_sig).j = 0;
		*val = d_A(coord_act.i, coord_act.j);
		*pos_ult_cero = 2;
	}
	else
	{
		*found = 0;
		itcount = 0;
		dif_i = iaux;
		dif_j = jaux;
		*val = 1;
		int pos = 0;
		while (*found == 0 && itcount <= 8) //starting from former pixel, look for nex pixel counterclockwise
		{
			counterclock_2(&dif_i, &dif_j, &iaux, &jaux, &pos);
			if (d_A(coord_act.i + iaux, coord_act.j + jaux) == 0)
			{
				*val = (*val)*pos; //storing in "val" the visited zeros
				if (pos > 1)
					*pos_ult_cero = pos;
			}

			if (d_A(coord_act.i + iaux, coord_act.j + jaux) != 0)
			{
				*found = 1;
				(*coord_sig).i = coord_act.i + iaux;
				(*coord_sig).j = coord_act.j + jaux;
			}
			else
			{
				dif_i = iaux;
				dif_j = jaux;
				itcount = itcount + 1;
			}
		}
	}
}

__device__ void rotate_later(
	byte *d_A, coord *coord_ant, coord *coord_sig, int *fin, int *val, int *pos_ult_cero, coord coord_act)
{
   //next rotations around a pixel, looking for other not covered triads;
    //line 13 , Algorithm 1 of paper
	int dif_i = (*coord_ant).i - coord_act.i;
	int dif_j = (*coord_ant).j - coord_act.j;
	int itcount, found, pos, iaux, jaux;
	*coord_sig = *coord_ant;
	found = 0;
	itcount = 0;
	*fin = 0;
	*val = 1;
	while (found == 0 && itcount <= 8) //look for zero pixel clockwise, starting from iant
	{
		if (dif_i*dif_j == 0) //one of the four  "cross " positions		
      {
			clockwise_2(&dif_i, &dif_j, &dif_i, &dif_j, &pos);
			if (d_A(coord_act.i + dif_i, coord_act.j + dif_j) != 0)
			{
				(*coord_sig).i = coord_act.i + dif_i;
				(*coord_sig).j = coord_act.j + dif_j;
			}
			clockwise_2(&dif_i, &dif_j, &iaux, &jaux, &pos);
		}
		else //one of the four  "corner " positions
		{
			clockwise_2(&dif_i, &dif_j, &iaux, &jaux, &pos);
		}
		if (d_A(coord_act.i + iaux, coord_act.j + jaux) != 0)
		{
			(*coord_sig).i = coord_act.i + iaux;
			(*coord_sig).j = coord_act.j + jaux;
		}
		if ((iaux == 0) && (jaux == -1))
		{
			found = 1;
			*fin = 1;
		}
		else if (d_A(coord_act.i + iaux, coord_act.j + jaux) == 0)
		{
			found = 1;
			// iant=i+iaux;
			// jant=j+jaux;
		}
		else
		{
			dif_i = iaux;
			dif_j = jaux;
			itcount = itcount + 1;
		}

	}

	*pos_ult_cero = pos;
	*val = (*val)*pos;
	dif_i = iaux;
	dif_j = jaux;
	if ((*fin == 0) && (found == 1))
	{
		found = 0;
		itcount = 0;
		while (found == 0 && itcount <= 8)
		{
			clockwise_2(&dif_i, &dif_j, &iaux, &jaux, &pos);
			if (d_A(coord_act.i + iaux, coord_act.j + jaux) == 0)
			{
				*val = (*val)*pos; //the visited "zeros" are accumulated in "pos"
				if (pos > 1)
				{
					*pos_ult_cero = pos;
				}
			}
			if ((iaux == 0) && (jaux == -1))
			{
				found = 1;
				(*coord_ant).i = coord_act.i + iaux;
				(*coord_ant).j = coord_act.j + jaux;
				if (d_A(coord_act.i + iaux, coord_act.j + jaux) == 0)
					*fin = 1;
				else if (d_A(coord_act.i + iaux, coord_act.j + jaux) != 0)
					*fin = 2;
			}
			else if (d_A(coord_act.i + iaux, coord_act.j + jaux) != 0)
			{
				found = 1;
				(*coord_ant).i = coord_act.i + iaux;
				(*coord_ant).j = coord_act.j + jaux;
			}
			else
			{
				dif_i = iaux;
				dif_j = jaux;
				itcount = itcount + 1;
			}
		}
	}
}





/**
 * Kernel CUDA: parallel_tracking
* kernel to be called using a single thread per block.
* Each thread (block) tracks the contours of its rectangle and stores them.
*Each thread visits all the pixels of its rectangle, if a contour pixel is found then it is tracked (using track_fw_bkw)
*and stored
*The "closed" contours (do not leave the rectangle) are directly stored in the global structure d_ind_conts_glob,
*using mutual exclusion to avoid data races
 *   
 */
__global__ void parallel_tracking(
	byte *d_A,
	byte *d_is_bord,
	int* d_numconts,
	VecCont* d_vec_conts,
	IndCont* d_ind_conts,
    IndCont* d_ind_conts_glob,      
     int* d_numconts_glob)
{
	//a thread per block

	int ib = blockIdx.x; //thread index
	int jb = blockIdx.y;
	int mb = gridDim.x;   //number of blocks
	int nb = gridDim.y;
	int numfbl = Mg / mb;  //dimensions of rectangles
	int numcbl = Ng / nb;

	// limits of rectangle
	int i_ini = (ib == 0) ? 1 : (ib * numfbl);
	int j_ini = (jb == 0) ? 1 : (jb * numcbl);
	int i_fin = (ib == mb - 1) ? (M - 1) : ((ib + 1) * numfbl - 1);
	int j_fin = (jb == nb - 1) ? (N - 1) : ((jb + 1) * numcbl - 1);

	// block index (from 0 to mfb*nfb-1)
	int indicebl = ib + mb * jb;

	int i_vec_conts = (jb)*numcbl*Mg * 2 - 1 + 2 * numfbl*numcbl*(ib);
	//The *2 is because the vector of points has size 2*Mg*Ng

	// initial position - 1 of the contours of present block in the structure d_ind_conts 
	int i_ind_conts = indicebl * MAX_N_BORDS - 1;

	coord coord_act, coord_ant, coord_sig;
	int found, val, pos_ult_cero;
	int borders_thispoint_tracked;
//main double loop to inspect all the points in the rectangle
	for (coord_act.j = j_ini; (coord_act.j <= j_fin); coord_act.j++)
	{
		for (coord_act.i = i_ini; (coord_act.i <= i_fin); coord_act.i++)
		{
			if (d_is_bord(coord_act.i, coord_act.j) > 0) //only the contour points are processed
			{
				rotate_ini(d_A, &coord_ant, &coord_sig, &found, &val, &pos_ult_cero, coord_act); //find first triad
				if (found != 0)
				{  //a triad has been found, not tracked yet
					if ((d_A(coord_act.i, coord_act.j) == 1) || (d_A(coord_act.i, coord_act.j) % pos_ult_cero) != 0)
					{
						d_A(coord_act.i, coord_act.j) *= val;
						i_ind_conts++; //new contour
						i_vec_conts++;
						d_numconts[indicebl]++;
						//d_vec_conts[i_vec_conts].act = coord_act;
						track_fw_bkw(&i_vec_conts, d_A, d_is_bord, d_numconts, d_vec_conts, d_ind_conts, i_ind_conts, i_ini, j_ini, i_fin, j_fin, coord_ant, coord_act, coord_sig);
						 if (d_ind_conts[i_ind_conts].sts==CLOSED_CONTOUR)
                                               {
                                               int contg=atomicAdd(d_numconts_glob,1);  //if closed contour, add to global structure
                                               d_ind_conts_glob[contg+1]=d_ind_conts[i_ind_conts];
                                               i_ind_conts--;
                                               d_numconts[indicebl]--;
                                               }
						borders_thispoint_tracked = 0;
						while (borders_thispoint_tracked == 0)
						{
							rotate_later(d_A, &coord_ant, &coord_sig, &borders_thispoint_tracked, &val, &pos_ult_cero, coord_act); //search for new triads not tracked
							if ((borders_thispoint_tracked != 1) && (d_A(coord_act.i, coord_act.j) % pos_ult_cero != 0))
							{
								d_A(coord_act.i, coord_act.j) *= val;
								i_vec_conts++;
								i_ind_conts++;
								d_numconts[indicebl]++;
								track_fw_bkw(&i_vec_conts, d_A,  d_is_bord, d_numconts, d_vec_conts, d_ind_conts, i_ind_conts, i_ini, j_ini, i_fin, j_fin, coord_ant, coord_act, coord_sig);
								if (d_ind_conts[i_ind_conts].sts==CLOSED_CONTOUR)
                                                              {
                                                               int contg=atomicAdd(d_numconts_glob,1);
                                                               d_ind_conts_glob[contg+1]=d_ind_conts[i_ind_conts];
                                                                i_ind_conts--;
                                                                d_numconts[indicebl]--;
                                                              }
								if (borders_thispoint_tracked == 2) //especial case
								{
									borders_thispoint_tracked = 0;
								}

							}
						}
					}
					else if (d_A(coord_act.i, coord_act.j) > 1)
					{
						borders_thispoint_tracked = 0;
						while (borders_thispoint_tracked == 0)
						{
							rotate_later(d_A, &coord_ant, &coord_sig, &borders_thispoint_tracked, &val, &pos_ult_cero, coord_act);
                         if ((borders_thispoint_tracked != 1) && (d_A(coord_act.i, coord_act.j) % pos_ult_cero != 0))
						
							{
								d_A(coord_act.i, coord_act.j) *= val;
								i_vec_conts++;
								i_ind_conts++;
								d_numconts[indicebl]++;
								track_fw_bkw(&i_vec_conts, d_A,  d_is_bord, d_numconts, d_vec_conts, d_ind_conts, i_ind_conts, i_ini, j_ini, i_fin, j_fin, coord_ant, coord_act, coord_sig);
								if (d_ind_conts[i_ind_conts].sts==CLOSED_CONTOUR)
                                                               {
                                                                 int contg=atomicAdd(d_numconts_glob,1);
                                                                 d_ind_conts_glob[contg+1]=d_ind_conts[i_ind_conts];
                                                                 i_ind_conts--;
                                                                 d_numconts[indicebl]--;
                                                               }
							}
						}
					}
				}
				else if ((d_is_bord(coord_act.i, coord_act.j) > 0) && (found == 0)) //contour of single point
				{
					i_vec_conts++;
					//i_ind_conts++;
					int contg=atomicAdd(d_numconts_glob,1);

					coord_ant.i = coord_act.i;
					coord_ant.j = coord_act.j;
					coord_sig.i = coord_act.i;
					coord_sig.j = coord_act.j;
					d_ind_conts_glob[contg+1].sts = CLOSED_CONTOUR;
					d_ind_conts_glob[contg+1].fin = i_vec_conts;
					d_ind_conts_glob[contg+1].ini = i_vec_conts;
					d_vec_conts[i_vec_conts].next = i_vec_conts;
					d_vec_conts[i_vec_conts].sig = coord_sig;
					d_vec_conts[i_vec_conts].act = coord_act;
					d_vec_conts[i_vec_conts].ant = coord_ant;

				}
			}
		}


	}
}

	 
		 
		 
	

/**
 * Kernel CUDA: Vertical_connection of borders . The closed borders are added to the global sructure
 *   Para cada bloque, creación de la lista de 
 */
__global__ void vertical_connection(

	int num_max_conts,
    int     *d_numconts,
    VecCont *d_vec_conts,
    IndCont *d_ind_conts,
	int     *d_numconts_out,
    IndCont *d_ind_conts_out,
	int *marked,
    int numfbl,
    int numcbl,
    IndCont* d_ind_conts_glob,      
    int* d_numconts_glob
	)
{

  int ib = blockIdx.x;
  int jb = blockIdx.y;
  int mb = 2*gridDim.x;
  int nb = gridDim.y;
  int i_con_ini, j_con_ini,i_antes, j_antes, i_salida, nump1,p2, jfc;
int i_next_in, j_next_in, i_next_out,j_next_out, pf_bueno;
int connected_border, ind_c_fuera;

  int ibloque, contorno, ind_b_fuera;
  int ibloque_arriba=ib*2+mb*(jb);
  int ibloque_sal_arriba=ibloque_arriba/2;
  //int ibloque_sal_arriba=ib+mb*(jb);
  int ibloque_abajo=(ib*2)+1+mb*(jb);
  int indice_ini=num_max_conts*(ibloque_arriba);
  int indice_ini_abajo=num_max_conts*(ibloque_abajo);
  int numcslocal=-1;
  int i_ind_conts=numcslocal+indice_ini;
  for (int cont=0; cont< d_numconts[ibloque_arriba];cont++)
    {  ibloque=ibloque_arriba;
	   int indcactual=cont+indice_ini;
       
            if (marked[indcactual]==0)	
              {
                 numcslocal++;
				 i_ind_conts++;
          
                 d_ind_conts_out[i_ind_conts]=d_ind_conts[indcactual];
                 int nump1=	d_ind_conts[indcactual].ini;
                 i_con_ini=d_vec_conts[nump1].act.i;
				 j_con_ini=d_vec_conts[nump1].act.j;
				 i_antes=d_vec_conts[nump1].ant.i;
				 j_antes=d_vec_conts[nump1].ant.j;
                 int fin=0;
                 contorno=cont;
                 i_salida=ibloque_abajo;
                 while(fin==0)
                   {  int indcaux=contorno+num_max_conts*(ibloque);
				   //  final point of contour
	                  int p2 = d_ind_conts[indcaux].fin;

	   // Coords of final point of contour, in present block
	                    int i_con_dentro = d_vec_conts[p2].act.i;
                        int j_con_dentro = d_vec_conts[p2].act.j;
	    // next point of contour, out of present block
	                   int i_con_fuera = d_vec_conts[p2].sig.i;
                       int j_con_fuera = d_vec_conts[p2].sig.j;
	    // Index of block where next pixel is
	                   int ibl_siguiente =  i_con_fuera / numfbl ;
	                 if ( ibl_siguiente > mb-1 )
	                   ibl_siguiente = mb-1;             
	                 int jbl_siguiente = j_con_fuera / numcbl ;
                    if ( jbl_siguiente > nb-1 )
                     jbl_siguiente = nb-1;

                    ind_b_fuera=ibl_siguiente+mb*(jbl_siguiente);
			        if (ind_b_fuera!=i_salida)
                        { fin =1; 
						 d_ind_conts_out[i_ind_conts].ini=nump1;
						 d_ind_conts_out[i_ind_conts].fin=p2;
                         d_ind_conts_out[i_ind_conts].sts=OPEN_CONTOUR;
                         marked[indcactual]=numcslocal+1;
						 }
					else
                         {	int connected_border=-1;
                            for ( jfc=0;jfc<  d_numconts[ind_b_fuera];jfc++)
                              {	 int ind_c_fuera=jfc+num_max_conts*(ind_b_fuera);
                                 int pf=d_ind_conts[ind_c_fuera].ini; 
                               i_next_in=d_vec_conts[pf].act.i;
                              j_next_in=d_vec_conts[pf].act.j;
                              i_next_out=d_vec_conts[pf].ant.i;
                              j_next_out=d_vec_conts[pf].ant.j;
                           if ((i_con_dentro==i_next_out)&&(j_con_dentro==j_next_out)&&(i_con_fuera==i_next_in)&&(j_con_fuera==j_next_in))
                             { connected_border=jfc; //jfc contour connects
                              pf_bueno=pf;
                              break;
                             }
                           }
                          if (connected_border==-1)

                          printf ("contorno no conectado:  bloquei %d, bloque j %d contorno %d, punto %d %d \n",ib*2, jb, jfc, i_con_dentro, j_con_dentro);
						  else
						    {
							   contorno=connected_border;
							   d_vec_conts[p2].next=pf_bueno;
							   if ((i_con_dentro==i_antes)&&(j_con_dentro==j_antes)&&(i_con_fuera==i_con_ini)&&(j_con_fuera==j_con_ini))
							     {
                             //conected with begin of contour, so that closes
                               fin=1; 
 
                                int contg=atomicAdd(d_numconts_glob,1); //store in global structure
                                d_ind_conts_glob[contg+1]=d_ind_conts[i_ind_conts];
                                i_ind_conts--;
                                numcslocal--;  
                                d_ind_conts_glob[contg+1].fin=p2;
                                d_ind_conts_glob[contg+1].ini=nump1;
                                d_ind_conts_glob[contg+1].sts=CLOSED_CONTOUR;
                                 marked[contorno+num_max_conts*(ind_b_fuera)]=-1;  
                                 marked[indcactual]=-1;
							   
                       			}	   
                             else if (marked[contorno+num_max_conts*(ind_b_fuera)]>0) //conecta con uno ya recorrido, pero no cierra
                               {fin=1;
                                 
                               int numcontorno=marked[contorno+num_max_conts*(ind_b_fuera)]-1; //este es el numero de contorno de salida
                               d_ind_conts_out[numcontorno+indice_ini].ini=nump1;
                               marked[indcactual]=numcontorno+1;

                               numcslocal--;
							   i_ind_conts--;
							   }
							   else
							   {  
                               int cont_siguiente= contorno+num_max_conts*(ind_b_fuera);                      
                                 marked[cont_siguiente]=numcslocal+1;
                       
 
                                 i_salida=ibloque;
                                 ibloque=ind_b_fuera;
								} 
                            }
                        }
					}
				}	
          //  }
     }
	 ibloque=ibloque_abajo;
     for (int cont=0; cont< d_numconts[ibloque_abajo];cont++)
         { ibloque=ibloque_abajo;
           int indcactual=cont+indice_ini_abajo;
        
                   if(marked[indcactual]==0) // if not marked as tracked, track it
                   {    
                    numcslocal=numcslocal+1;
					i_ind_conts++;
            
                    d_ind_conts_out[i_ind_conts]=d_ind_conts[indcactual];
                    nump1= d_ind_conts[indcactual].ini;  
                    p2=d_ind_conts[indcactual].fin;
  	                    int i_con_dentro = d_vec_conts[p2].act.i;
                        int j_con_dentro = d_vec_conts[p2].act.j;
	    // next pixel, out of present block
	                   int i_con_fuera = d_vec_conts[p2].sig.i;
                       int j_con_fuera = d_vec_conts[p2].sig.j;
                 
                    int ibl_fuera=i_con_fuera/numfbl; //coords of block of next pixel de bloque del punto siguiente
                        if (ibl_fuera>mb-1)
						 ibl_fuera=mb-1;
                    int jbl_fuera=j_con_fuera/numcbl; 
                          if (jbl_fuera>nb-1)
						  jbl_fuera=nb-1;
                    int ind_b_fuera=ibl_fuera+mb*(jbl_fuera);
	                if (ind_b_fuera!=ibloque_arriba) //leaves by another block, added to list of open contours                                               
                        { d_ind_conts_out[i_ind_conts].ini=nump1;
                         d_ind_conts_out[i_ind_conts].fin=p2;
                         d_ind_conts_out[i_ind_conts].sts=OPEN_CONTOUR; 
						 }
                  else   				
					   {connected_border=-1;
                        for (jfc=0;jfc< d_numconts[ind_b_fuera];jfc++)
                              {	 ind_c_fuera=jfc+num_max_conts*(ind_b_fuera);
                                 int pf=d_ind_conts[ind_c_fuera].ini; 
                               i_next_in=d_vec_conts[pf].act.i;
                              j_next_in=d_vec_conts[pf].act.j;
                              i_next_out=d_vec_conts[pf].ant.i;
                              j_next_out=d_vec_conts[pf].ant.j;
                           if ((i_con_dentro==i_next_out)&&(j_con_dentro==j_next_out)&&(i_con_fuera==i_next_in)&&(j_con_fuera==j_next_in))
                             { connected_border=jfc; //jfc contour connects
                              pf_bueno=pf;
                              break;
                             }
                           }
					                          
                          if (connected_border==-1)
                         
                          printf ("contorno no conectado:  bloquei %d, bloque j %d contorno %d, punto %d %d \n",ib*2, jb, jfc, i_con_dentro, j_con_dentro);
						  else
						    {
							   contorno=connected_border;
							   d_vec_conts[p2].next=pf_bueno;

                            int numcontorno=marked[contorno+num_max_conts*(ind_b_fuera)]-1; 
							d_ind_conts_out[numcontorno+indice_ini].ini=nump1; 

                               marked[indcactual]=numcontorno+1;


                               numcslocal--;
							   i_ind_conts--;

                           }
						}   
				}
            //}
        }
		d_numconts_out[ibloque_sal_arriba]=numcslocal+1;
    }
	
  /**
 * Kernel CUDA: horizontal_connection of borders . The closed borders are added to the global sructure
 *    
 */
__global__ void horizontal_connection(
	int num_max_conts,
    int     *d_numconts,
    VecCont *d_vec_conts,
    IndCont *d_ind_conts,
	int     *d_numconts_out,
    IndCont *d_ind_conts_out,
	int *marked,
    int numfbl,
    int numcbl,
    IndCont* d_ind_conts_glob,      
    int* d_numconts_glob
	)
{ //When this kernel is called , there will be only a row of blocks 

  //int ib = blockIdx.x;
  int jb = blockIdx.y;
  int mb = gridDim.x;
  int nb = 2*gridDim.y;
  int i_con_ini, j_con_ini,i_antes, j_antes, i_salida, nump1,p2, jfc;
int i_next_in, j_next_in, i_next_out,j_next_out, pf_bueno;
int connected_border, ind_c_fuera, contorno,ind_b_fuera;
  int ibloque;
  int ibloque_izquierda=2*jb;
  int ibloque_sal_izquierda=jb;
  int ibloque_derecha=2*jb+1;
  int indice_ini=num_max_conts*(ibloque_izquierda);
  int indice_ini_derecha=num_max_conts*(ibloque_derecha);
  int numcslocal=-1;
  int i_ind_conts=numcslocal+indice_ini;
  for (int cont=0; cont< d_numconts[ibloque_izquierda];cont++)
    {  ibloque=ibloque_izquierda;
	   int indcactual=cont+indice_ini;
       
            if (marked[indcactual]==0	)
              {
                 numcslocal++;
				 i_ind_conts++;
                 
                 d_ind_conts_out[i_ind_conts]=d_ind_conts[indcactual];
                 int nump1=	d_ind_conts[indcactual].ini;
                 i_con_ini=d_vec_conts[nump1].act.i;
				 j_con_ini=d_vec_conts[nump1].act.j;
				 i_antes=d_vec_conts[nump1].ant.i;
				 j_antes=d_vec_conts[nump1].ant.j;
                 
                 int fin=0;
                 contorno=cont;
                 i_salida=ibloque_derecha;
                 while(fin==0)
                   {  int indcaux=contorno+num_max_conts*(ibloque);
				   // final point of contour
	                    p2 = d_ind_conts[indcaux].fin;

	   // Coords of final point of contour, in present rectangle
	                    int i_con_dentro = d_vec_conts[p2].act.i;
                        int j_con_dentro = d_vec_conts[p2].act.j;
	    // next point of contour, out of present rectangle 
	                   int i_con_fuera = d_vec_conts[p2].sig.i;
                       int j_con_fuera = d_vec_conts[p2].sig.j;
	    // Index of rectangle where next pixel is; in horizontal, ibl_siguiente is always zero
	                     
	                 int jbl_siguiente = j_con_fuera / numcbl ;
                    if ( jbl_siguiente > nb-1 )
                     jbl_siguiente = nb-1;
                 ind_b_fuera=mb*(jbl_siguiente);//indice de bloque siguiente
                         int jfc;
			        if (ind_b_fuera!=i_salida)
                        { fin =1; 
						 d_ind_conts_out[i_ind_conts].ini=nump1;
						 d_ind_conts_out[i_ind_conts].fin=p2;
                         d_ind_conts_out[i_ind_conts].sts=OPEN_CONTOUR;
                         marked[indcactual]=numcslocal+1;
						 }
					else
                         {	int connected_border=-1;
               
                            for (jfc=0;jfc<  d_numconts[ind_b_fuera];jfc++)
                              {	 ind_c_fuera=jfc+num_max_conts*(ind_b_fuera);
                                 int pf=d_ind_conts[ind_c_fuera].ini; 
                               i_next_in=d_vec_conts[pf].act.i;
                              j_next_in=d_vec_conts[pf].act.j;
                              i_next_out=d_vec_conts[pf].ant.i;
                              j_next_out=d_vec_conts[pf].ant.j;
  
                           if ((i_con_dentro==i_next_out)&&(j_con_dentro==j_next_out)&&(i_con_fuera==i_next_in)&&(j_con_fuera==j_next_in))
                             { connected_border=jfc; //jfc contour connects
                              pf_bueno=pf;
                              break;
                             }
                           }
                 
                          if (connected_border==-1)
                         
{printf ("contorno no conectado:  bloquei %d, bloque j %d contorno %d, punto %d %d \n",1, jb, jfc, i_con_dentro, j_con_dentro);}
						  else
						    {
                      
							   contorno=connected_border;
							   d_vec_conts[p2].next=pf_bueno;
							   if ((i_con_dentro==i_antes)&&(j_con_dentro==j_antes)&&(i_con_fuera==i_con_ini)&&(j_con_fuera==j_con_ini))
							     {
                             //connects with start of contour, so that closes
                               fin=1; //connected with end
                                 int contg=atomicAdd(d_numconts_glob,1);
                                d_ind_conts_glob[contg+1]=d_ind_conts[i_ind_conts];
                                i_ind_conts--;
                                numcslocal--;  
                                d_ind_conts_glob[contg+1].fin=p2;
                                d_ind_conts_glob[contg+1].ini=nump1;
                                d_ind_conts_glob[contg+1].sts=CLOSED_CONTOUR;
                                 marked[contorno+num_max_conts*(ind_b_fuera)]=-1;  
                                 marked[indcactual]=-1;       
                               
                       			}	   
                             else if (marked[contorno+num_max_conts*(ind_b_fuera)]>0) //connects with contour already tracked, but does not close
                               {fin=1;
                               int numcontorno=marked[contorno+num_max_conts*(ind_b_fuera)]-1; 
                               d_ind_conts_out[numcontorno+indice_ini].ini=nump1;
                               marked[indcactual]=numcontorno+1;


//El contorno i_ind_conts se ha integrado en el numcontorno, se disminuyen numcslocal y i_ind_conts

                               numcslocal--;
							   i_ind_conts--;
							   }
							   else
							   {  
            int cont_siguiente= contorno+num_max_conts*(ind_b_fuera);                                          
                                 marked[cont_siguiente]=numcslocal+1;   
   
                                 i_salida=ibloque;
                                 ibloque=ind_b_fuera;
								} 
                            }
                        }
					}
				}	
         //   }
     }
	 ibloque=ibloque_derecha;
     for (int cont=0; cont< d_numconts[ibloque_derecha];cont++)
         { ibloque=ibloque_derecha;
           int indcactual=cont+indice_ini_derecha;
       
                   if(marked[indcactual]==0) // if not marked as tracked, track it
                   {    
                    numcslocal=numcslocal+1;
					i_ind_conts++;
                   d_ind_conts_out[i_ind_conts]=d_ind_conts[indcactual];
                    nump1= d_ind_conts[indcactual].ini; 
                    p2=d_ind_conts[indcactual].fin;
  	                    int i_con_dentro = d_vec_conts[p2].act.i;
                        int j_con_dentro = d_vec_conts[p2].act.j;
	    // next point, out of rectangle
	                   int i_con_fuera = d_vec_conts[p2].sig.i;
                       int j_con_fuera = d_vec_conts[p2].sig.j;
                 
                    
                    int jbl_fuera=j_con_fuera/numcbl; 
                          if (jbl_fuera>nb-1)
						  jbl_fuera=nb-1;
                    int ind_b_fuera=mb*(jbl_fuera);
	                if (ind_b_fuera!=ibloque_izquierda)                                                
                        { d_ind_conts_out[i_ind_conts].ini=nump1;
                         d_ind_conts_out[i_ind_conts].fin=p2;
                         d_ind_conts_out[i_ind_conts].sts=OPEN_CONTOUR; 
						 }
                  else   				
					   {connected_border=-1;
                        for (jfc=0;jfc< d_numconts[ind_b_fuera];jfc++)
                              {	 ind_c_fuera=jfc+num_max_conts*(ind_b_fuera);
                                 int pf=d_ind_conts[ind_c_fuera].ini; 
                               i_next_in=d_vec_conts[pf].act.i;
                              j_next_in=d_vec_conts[pf].act.j;
                              i_next_out=d_vec_conts[pf].ant.i;
                              j_next_out=d_vec_conts[pf].ant.j;
                           if ((i_con_dentro==i_next_out)&&(j_con_dentro==j_next_out)&&(i_con_fuera==i_next_in)&&(j_con_fuera==j_next_in))
                             { connected_border=jfc; //jfc contour connects
                               pf_bueno=pf;
                              break;
                             }
                           }
					                          
                          if (connected_border==-1)
                          
                          printf ("contorno no conectado:  bloquei %d, bloque j %d contorno %d, punto %d %d \n",2, jb, jfc, 0, j_con_dentro);
						  else
						    {
							   contorno=connected_border;
							   d_vec_conts[p2].next=pf_bueno;

                            int numcontorno=marked[contorno+num_max_conts*(ind_b_fuera)]-1; 
                            
							d_ind_conts_out[numcontorno+num_max_conts*(ind_b_fuera)].ini=nump1;
                               marked[indcactual]=numcontorno+1;


                               numcslocal--;
							   i_ind_conts--;
  
                            

                           }
						}   
				}
          //  }
        }
		d_numconts_out[ibloque_sal_izquierda]=numcslocal+1;
    }


/**
 * plot contours obtained in a new image, GPU
 */
__global__ void plot_contours_gpu(
    IndCont *d_ind_conts,
    VecCont *d_vec_conts,
    byte *d_Asal)
{
  int i = blockIdx.x;
  
    // Position and coordinates of initial point
    int i_ind_conts_ini = d_ind_conts[i].ini;
    coord p = d_vec_conts[i_ind_conts_ini].act;

    // Position of final point
    int i_ind_conts_fin = d_ind_conts[i].fin;

    // write contour number in starting point
    d_Asal(p.i, p.j) = i + 1;

    // follow next points
    while ( i_ind_conts_ini != i_ind_conts_fin )
    {
      i_ind_conts_ini = d_vec_conts[i_ind_conts_ini].next;
      p = d_vec_conts[i_ind_conts_ini].act;
      d_Asal(p.i, p.j) = i + 1;
    }
  
}


// /**
//  * plot contours obtained in a new image, CPU
//  */
// void plot_contours(
//     int numcsg,
//     IndCont *h_ind_conts,
//     VecCont *h_vec_conts,
//     byte *h_Asal)
// {
//   for( int i = 0; i < numcsg; i++ )
//   {
//     // Position and coordinates of initial point
//     int i_ind_conts_ini = h_ind_conts[i].ini;
//     coord p = h_vec_conts[i_ind_conts_ini].act;

//     // Position of final point
//     int i_ind_conts_fin = h_ind_conts[i].fin;

//     // write contour number in starting point
//     h_Asal(p.i, p.j) = i + 1;

//     // follow next points
//     while ( i_ind_conts_ini != i_ind_conts_fin )
//     {
//       i_ind_conts_ini = h_vec_conts[i_ind_conts_ini].next;
//       p = h_vec_conts[i_ind_conts_ini].act;
//       h_Asal(p.i, p.j) = i + 1;
//     }
//   }
// }

//copy image to extended format
void copy_p_a_g(byte *h_A, byte *h_Ag)
{


  for( int j = 0; j < N; j++ )
  {
    for( int i = 0; i < M; i++ )
    {
      h_Ag(i, j) = h_A(i,j);

    }
  }
}

// //copy image form extended format to original format
// void copy_g_a_p(byte *h_Ag, byte *h_A)
// {


//   for( int j = 0; j < N; j++ )
//   {
//     for( int i = 0; i < M; i++ )
//     {
//       h_A(i, j) = h_Ag(i,j);

//     }
//   }
// }

// /**
//  * Read image from .bin file 
//  */
// void read4bin (byte *h_A, const char fname[])
// {
//   byte *h_Asal = (byte*) malloc( M * N * sizeof(byte) );

//   FILE *fd;
//   fd = fopen(fname, "r" );
//   fread( (char*) h_Asal, sizeof(byte), M * N, fd );
//   fclose( fd );
//   int k=0;
//   for( int j = 0; j < N; j++ )
//   {
//     for( int i = 0; i < M; i++ )
//     {
//       h_A(i, j) = h_Asal[k];
//       k++;
//     }
//   }
// }


// /**
//  * output matrix to .bin file
//  */
// void write2bin (byte *h_A, const char fname[])
// {
//   byte *h_Asal = (byte*) malloc( M * N * sizeof(byte) );
//   int k=0;
//   for( int j = 0; j < N; j++ )
//   {
//     for( int i = 0; i < M; i++ )
//     {
//       h_Asal[k++] = h_A(i, j);
//     }
//   }

//   FILE *fd;
//   fd = fopen(fname, "w" );
//   fwrite( (char*) h_Asal, sizeof(byte), M * N, fd );
//   fclose(fd);
// }


// /**
//  * Output from small matrix to screen */
// void printMatrix(byte *h_A, const char info[])
// {
//   if ( (M < 20) && (N < 20) )
//   {
//     printf("%s:\n   ", info);
//     for( int j = 0; j < N; j++ )
//       printf("%2d  ", j);
//     printf("\n");
//     for( int i = 0; i < M; i++ )
//     {
//       printf("%d   ", i);
//       for( int j = 0; j < N; j++ )
//       {
// 	printf("%d   ", h_A(i,j));
//       }
//       printf("\n");
//     }
//   }
// }


// /**
//  * Output from matrix .bin fila and to screen 
//  */
// void outputMatrix (byte *h_A, const char name[])
// {
//   printMatrix(h_A, name);
//   write2bin(h_A, name);
// }

//kernel to ensure that outer limit of image is full of zeros
__global__ void borde_ceros(
                byte *d_A
        )
{
        int i = threadIdx.x;
    while(i<M)
     {
         d_A[i]=0;
        d_A(i ,(N - 1)) = 0;
         i=i+1024;
     } 
   i = threadIdx.x;
    while(i<N)
     {

        d_A(0 ,i) = 0;
        d_A(M-1 ,i) = 0;
         i=i+1024;
     }
 
}







/**
 * routines for GPU timing
 */
// cudaEvent_t start, stop;
// float gpu_ElapsedTime;
// float gpu_ElapsedTimes[8][1000];

// void startCudaTimer(void)
// {
//   cudaEventRecord(start, 0);
// }

// void stopCudaTimer(const char* text)
// {
//   cudaEventRecord(stop, 0);
//   cudaEventSynchronize(stop);
//   cudaEventElapsedTime(&gpu_ElapsedTime, start, stop);
//   printf("%s: %f ms (GPU)\n", text, gpu_ElapsedTime);
// }

// void saveCudaTimer(int step, int run)
// {
//   cudaEventRecord(stop, 0);
//   cudaEventSynchronize(stop);
//   cudaEventElapsedTime(&gpu_ElapsedTimes[step][run], start, stop);
// }

// void averageCudaTimer(int step, int runs, const char* text)
// {
//   float gpu_ElapsedTime = 0;
//   // If there are more than one execution, discard the first one
//   for( int run = (runs > 1 ? 1 : 0); run < runs; run++ ) 
//   {
//     gpu_ElapsedTime += gpu_ElapsedTimes[step][run];
//   }
//   if ( runs == 1)
//   {
//     printf("%s: %f ms (GPU)\n", text, gpu_ElapsedTime );
//   }
//   else
//   {
//     printf("%s: %f ms (GPU average of %d executions)\n", text, gpu_ElapsedTime / (runs - 1), runs - 1 );
//   }
// }

