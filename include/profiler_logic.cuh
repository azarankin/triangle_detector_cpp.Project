#pragma once

#ifdef ENABLE_NVTX
    #include <nvtx3/nvToolsExt.h>
    #define PROFILER_PUSH(name) nvtxRangePushA(name)
    #define PROFILER_POP() nvtxRangePop()
#else
    #define PROFILER_PUSH(name)
    #define PROFILER_POP()
#endif

#ifdef ENABLE_PROFILING
    #include <cuda_runtime.h>
    //#include <iostream>

    struct CudaEventProfiler {
        cudaEvent_t start, stop;
        float elapsed_ms = 0.0f;

        CudaEventProfiler() {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
        }

        ~CudaEventProfiler() {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        void tic(cudaStream_t stream = 0) {
            cudaEventRecord(start, stream);
        }

        float toc(cudaStream_t stream = 0) {
            cudaEventRecord(stop, stream);
            cudaStreamSynchronize(stream); // Wait for the stream to finish
            cudaEventElapsedTime(&elapsed_ms, start, stop);
            return elapsed_ms;
        }
    };
#endif

// Inline functions for profiling
inline void profiler_push(const char* name) { PROFILER_PUSH(name); }
inline void profiler_pop() { PROFILER_POP(); }



// #ifdef ENABLE_PROFILING
//     CudaEventProfiler prof;
//     prof.tic();
// #endif

// // ... Kernel/Code ...

// #ifdef ENABLE_PROFILING
//     float ms = prof.toc();
//     //std::cout << "Kernel time: " << ms << " ms\n";
// #endif