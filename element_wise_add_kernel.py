import torch
from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    #   float4 v_ = *(float4*)(A + (((((((int)blockIdx.y) * 131072) + (i * 2048)) + ((((int)threadIdx.x) >> 6) * 1024)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) & 63) * 4)));
    #   float4 v__1 = *(float4*)(B + (((((((int)blockIdx.y) * 131072) + (i * 2048)) + ((((int)threadIdx.x) >> 6) * 1024)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) & 63) * 4)));
    # *(uint2*)(C + (((((((int)blockIdx.y) * 131072) + (i * 2048)) + ((((int)threadIdx.x) >> 6) * 1024)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) & 63) * 4))) 
    for i in cutlass.range_constexpr(64):
        offset_A = bidy * 131072 + i * 2048 + (tidx >> 6) * 1024 + bidx * 256 + (tidx & 63) * 4
        offset_B = bidy * 131072 + i * 2048 + (tidx >> 6) * 1024 + bidx * 256 + (tidx & 63) * 4
        offset_C = bidy * 131072 + i * 2048 + (tidx >> 6) * 1024 + bidx * 256 + (tidx & 63) * 4

        offset_A = cute.assume(offset_A, divby = 4)
        offset_B = cute.assume(offset_B, divby = 4)
        offset_C = cute.assume(offset_C, divby = 4)

        tA = cute.make_tensor(gA.iterator + offset_A, (4))
        tB = cute.make_tensor(gB.iterator + offset_B, (4))
        tC = cute.make_tensor(gC.iterator + offset_C, (4))

        tC.store((tA.load() + tB.load()).to(cute.Float16))

@cute.jit
def elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor
):
    kernel = elementwise_add_kernel(mA, mB, mC)
    kernel.launch(grid=(4,4,1),
                  block=(128, 1, 1))

M, N = 512, 1024

a = torch.randn(M, N, device="cuda", dtype=torch.float32)
b = torch.randn(M, N, device="cuda", dtype=torch.float32)
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)

# Compile kernel
elementwise_add_ = cute.compile(elementwise_add, a_, b_, c_)
elementwise_add_(a_, b_, c_)

# verify correctness
torch.testing.assert_close(c, (a + b).to(torch.float16))
print("pass!")