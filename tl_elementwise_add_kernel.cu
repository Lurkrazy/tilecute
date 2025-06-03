#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>

extern "C" __global__ void main_kernel(float* __restrict__ A, float* __restrict__ B, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(float* __restrict__ A, float* __restrict__ B, half_t* __restrict__ C) {
  #pragma unroll
  for (int i = 0; i < 64; ++i) {
    uint2 __1;
    float4 __2;
      float4 v_ = *(float4*)(A + (((((((int)blockIdx.y) * 131072) + (i * 2048)) + ((((int)threadIdx.x) >> 6) * 1024)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) & 63) * 4)));
      float4 v__1 = *(float4*)(B + (((((((int)blockIdx.y) * 131072) + (i * 2048)) + ((((int)threadIdx.x) >> 6) * 1024)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) & 63) * 4)));
      __2.x = (v_.x+v__1.x);
      __2.y = (v_.y+v__1.y);
      __2.z = (v_.z+v__1.z);
      __2.w = (v_.w+v__1.w);
    ((half2*)(&(__1.x)))->x = (half_t)(__2.x);
    ((half2*)(&(__1.x)))->y = (half_t)(__2.y);
    ((half2*)(&(__1.y)))->x = (half_t)(__2.z);
    ((half2*)(&(__1.y)))->y = (half_t)(__2.w);
    *(uint2*)(C + (((((((int)blockIdx.y) * 131072) + (i * 2048)) + ((((int)threadIdx.x) >> 6) * 1024)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) & 63) * 4))) = __1;
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    return 0;
}

extern "C" int call(float* __restrict__ A, float* __restrict__ B, half_t* __restrict__ C, cudaStream_t stream=cudaStreamDefault) {
	main_kernel<<<dim3(4, 4, 1), dim3(128, 1, 1), 0, stream>>>(A, B, C);
	TILELANG_CHECK_LAST_ERROR("main_kernel");

	return 0;
}
