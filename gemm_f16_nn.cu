#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>

extern "C" __global__ void main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  half_t C_local[128];
  #pragma unroll
  for (int i = 0; i < 64; ++i) {
    *(uint1*)(C_local + (i * 2)) = make_uint1(__pack_half2(half_t(0.000000e+00f), half_t(0.000000e+00f)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((i_1 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), A+((((((int)blockIdx.y) * 98304) + (i_1 * 24576)) + ((((int)threadIdx.x) >> 2) * 768)) + ((((int)threadIdx.x) & 3) * 8)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), B+((((i_2 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 23; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      tl::cp_async_gs<16>(buf_dyn_shmem+(((((((k + 1) & 1) * 8192) + (i_3 * 2048)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), A+((((((((int)blockIdx.y) * 98304) + (i_3 * 24576)) + ((((int)threadIdx.x) >> 2) * 768)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      tl::cp_async_gs<16>(buf_dyn_shmem+((((((((((k + 1) & 1) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), B+((((((k * 32768) + (i_4 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 32768));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<1>();
    __syncthreads();
    tl::gemm_ss<128, 128, 32, 2, 2, 0, 0, 0>((&(((half_t*)buf_dyn_shmem)[((k & 1) * 4096)])), (&(((half_t*)buf_dyn_shmem)[(((k & 1) * 4096) + 8192)])), (&(C_local[0])));
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<128, 128, 32, 2, 2, 0, 0, 0>((&(((half_t*)buf_dyn_shmem)[4096])), (&(((half_t*)buf_dyn_shmem)[12288])), (&(C_local[0])));
  #pragma unroll
  for (int i_5 = 0; i_5 < 64; ++i_5) {
    *(uint1*)(C + (((((((((((int)blockIdx.y) * 131072) + (((i_5 & 7) >> 1) * 32768)) + (((((int)threadIdx.x) & 63) >> 5) * 16384)) + ((i_5 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.x) * 128)) + ((i_5 >> 3) * 16)) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(C_local + (i_5 * 2));
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_main_kernel = cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);
    if (result_main_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 32768, cudaGetErrorString(result_main_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C, cudaStream_t stream=cudaStreamDefault) {
	main_kernel<<<dim3(8, 4, 1), dim3(128, 1, 1), 32768, stream>>>(A, B, C);
	TILELANG_CHECK_LAST_ERROR("main_kernel");

	return 0;
}
