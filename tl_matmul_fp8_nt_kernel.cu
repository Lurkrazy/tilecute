#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, float* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[128];
  __shared__ uint64_t _mbarrier[6];
  if (((int)threadIdx.x) == 0) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
    tl::mbarrier_init(_mbarrier[0], 128);
    tl::mbarrier_init(_mbarrier[1], 128);
    tl::mbarrier_init(_mbarrier[2], 128);
    tl::mbarrier_init(_mbarrier[3], 128);
    tl::mbarrier_init(_mbarrier[4], 128);
    tl::mbarrier_init(_mbarrier[5], 128);
  }
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    for (int k = 0; k < 16; ++k) {
      tl::mbarrier_wait(_mbarrier[((k % 3) + 3)], (((k % 6) / 3) ^ 1));
      if (((int)threadIdx.x) == 128) {
        tl::mbarrier_expect_tx(_mbarrier[(k % 3)], 8192);
        tl::tma_load(A_desc, _mbarrier[(k % 3)], (&(((fp8_e4_t*)buf_dyn_shmem)[((k % 3) * 8192)])), (k * 64), (((int)blockIdx.y) * 128));
        tl::mbarrier_expect_tx(_mbarrier[(k % 3)], 8192);
        tl::tma_load(B_desc, _mbarrier[(k % 3)], (&(((fp8_e4_t*)buf_dyn_shmem)[(((k % 3) * 8192) + 24576)])), (k * 64), (((int)blockIdx.x) * 128));
      }
      tl::mbarrier_arrive(_mbarrier[(k % 3)]);
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
      *(float2*)(C_local + (i * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
    }
    tl::fence_proxy_async();
    for (int k_1 = 0; k_1 < 16; ++k_1) {
      tl::mbarrier_wait(_mbarrier[(k_1 % 3)], ((k_1 % 6) / 3));
      tl::gemm_ss<128, 128, 64, 4, 1, 0, 1, 0, true>((&(((fp8_e4_t*)buf_dyn_shmem)[((k_1 % 3) * 8192)])), (&(((fp8_e4_t*)buf_dyn_shmem)[(((k_1 % 3) * 8192) + 24576)])), (&(C_local[0])));
      tl::mbarrier_arrive(_mbarrier[((k_1 % 3) + 3)]);
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 64; ++i_1) {
      *(float2*)(C + ((((((((((int)blockIdx.y) * 131072) + ((i_1 >> 5) * 65536)) + ((((int)threadIdx.x) >> 5) * 16384)) + ((i_1 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.x) * 128)) + (((i_1 & 31) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(C_local + (i_1 * 2));
    }
  }
}


#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {
    return error_buf;
}

extern "C" int init() {
    error_buf[0] = '\0';
    
    cudaError_t result_main_kernel = cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
    if (result_main_kernel != CUDA_SUCCESS) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", 49152, cudaGetErrorString(result_main_kernel));
        return -1;
    }

    return 0;
}

extern "C" int call(fp8_e4_t* __restrict__ A, fp8_e4_t* __restrict__ B, float* __restrict__ C, cudaStream_t stream=cudaStreamDefault) {

	CUtensorMap A_desc;
	CUtensorMapDataType A_desc_type= (CUtensorMapDataType)0;
	cuuint32_t A_desc_tensorRank= 2;
	void *A_desc_globalAddress= A;
	cuuint64_t A_desc_globalDim[2]= {1024,1024};
	cuuint64_t A_desc_globalStride[2]= {1,1024};
	cuuint32_t A_desc_boxDim[2]= {64,128};
	cuuint32_t A_desc_elementStrides[2]= {1,1};
	CUtensorMapInterleave A_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle A_desc_swizzle= (CUtensorMapSwizzle)2;
	CUtensorMapL2promotion A_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill A_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult A_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &A_desc, A_desc_type, A_desc_tensorRank, A_desc_globalAddress, A_desc_globalDim, A_desc_globalStride + 1, A_desc_boxDim, A_desc_elementStrides, A_desc_interleave, A_desc_swizzle, A_desc_l2Promotion, A_desc_oobFill);

	if (A_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor A_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}

	CUtensorMap B_desc;
	CUtensorMapDataType B_desc_type= (CUtensorMapDataType)0;
	cuuint32_t B_desc_tensorRank= 2;
	void *B_desc_globalAddress= B;
	cuuint64_t B_desc_globalDim[2]= {1024,1024};
	cuuint64_t B_desc_globalStride[2]= {1,1024};
	cuuint32_t B_desc_boxDim[2]= {64,128};
	cuuint32_t B_desc_elementStrides[2]= {1,1};
	CUtensorMapInterleave B_desc_interleave= (CUtensorMapInterleave)0;
	CUtensorMapSwizzle B_desc_swizzle= (CUtensorMapSwizzle)2;
	CUtensorMapL2promotion B_desc_l2Promotion= (CUtensorMapL2promotion)2;
	CUtensorMapFloatOOBfill B_desc_oobFill= (CUtensorMapFloatOOBfill)0;

	CUresult B_desc_result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &B_desc, B_desc_type, B_desc_tensorRank, B_desc_globalAddress, B_desc_globalDim, B_desc_globalStride + 1, B_desc_boxDim, B_desc_elementStrides, B_desc_interleave, B_desc_swizzle, B_desc_l2Promotion, B_desc_oobFill);

	if (B_desc_result != CUDA_SUCCESS) {
		std::stringstream ss;
		ss << "Error: Failed to initialize the TMA descriptor B_desc";
		snprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
		return -1;
	}
	main_kernel<<<dim3(8, 8, 1), dim3(256, 1, 1), 49152, stream>>>(A_desc, B_desc, C);
	TILELANG_CHECK_LAST_ERROR("main_kernel");

	return 0;
}
