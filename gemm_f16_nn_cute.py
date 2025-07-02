import torch
from functools import partial
from typing import Tuple, Type
import math
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils.ampere_helpers as ampere_utils
from cutlass.cute.runtime import from_dlpack

from utils import *

cta_tiler = (128, 128, 32)
# M*K @ K*N, R @ C
# in Cute, K major is row_major
a_major_mode = cutlass.utils.LayoutEnum.ROW_MAJOR
b_major_mode = cutlass.utils.LayoutEnum.COL_MAJOR


def _make_gmem_tiled_copy_AB(atom_copy, dtype, major_mode, copy_bits):
    copy_elems = copy_bits // dtype.width
    shape_dim_1 = cute.size(32) // copy_elems
    # thread layout for copy
    thread_layout = cute.make_layout(
        (128 // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
    )
    if major_mode != cutlass.utils.LayoutEnum.ROW_MAJOR:
        shape_dim_0 = cute.size(128) // copy_elems
        thread_layout = cute.make_layout(
            (shape_dim_0, 128 // shape_dim_0), stride=(1, shape_dim_0)
        )
    # Value layout for copy
    value_layout = (
        cute.make_layout((1, copy_elems))
        if major_mode == cutlass.utils.LayoutEnum.ROW_MAJOR
        else cute.make_layout((copy_elems, 1))
    )
    return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

def _make_smem_layout_AB(dtype, major_mode, copy_bits, smem_tiler):
    # print(smem_tiler)
    is_row_major = major_mode == cutlass.utils.LayoutEnum.ROW_MAJOR
    major_mode_size = (
        smem_tiler[1] if is_row_major else smem_tiler[0]
    )
    major_mode_size = 64 if major_mode_size >= 64 else major_mode_size

    swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
    swizzle_bits = min(swizzle_bits, 3)

    layout_atom_outer = (
        cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
        if is_row_major
        else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
    )
    # print(layout_atom_outer)
    layout_atom = cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, 3, 3),
        0,
        layout_atom_outer,
    )
    layout = cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2) if is_row_major else (1, 0, 2))
    return layout

@cute.jit
def mark_align(A: cute.Tensor, align: int):
    return cute.make_tensor(A.iterator.align(align), A.layout)

@cute.jit
def gemm_ss(
    sA: cute.Tensor,
    sB: cute.Tensor,
    rC: cute.Tensor,
    tiled_mma: cute.TiledMma,
):
    tidx, _, _ = cute.arch.thread_idx()
    thr_mma = tiled_mma.get_slice(tidx)

    sA = mark_align(sA, 16)
    sB = mark_align(sB, 16)
    
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    
    tCrA = tiled_mma.make_fragment_A(tCsA)
    tCrB = tiled_mma.make_fragment_B(tCsB)
    tCrC = cute.make_tensor(rC.iterator, tiled_mma.partition_shape_C((cta_tiler[0], cta_tiler[1])))

    # Create the copy atoms for the copy from shared memory to register
    atom_copy_s2r_A = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(a_major_mode != cutlass.utils.LayoutEnum.ROW_MAJOR, 4),
        sA.element_type,
    )
    atom_copy_s2r_B = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(b_major_mode != cutlass.utils.LayoutEnum.ROW_MAJOR, 4),
        sB.element_type,
    )

    # Creates the tiled copy so that it matches the thread-value layout
    # expected by the tiled mma
    tiled_copy_s2r_A = cute.make_tiled_copy(
        atom_copy_s2r_A,
        layout_tv=tiled_mma.tv_layout_A_tiled,
        tiler_mn=(tiled_mma.get_tile_size(0), tiled_mma.get_tile_size(2)),
    )
    tiled_copy_s2r_B = cute.make_tiled_copy(
        atom_copy_s2r_B,
        layout_tv=tiled_mma.tv_layout_B_tiled,
        tiler_mn=(tiled_mma.get_tile_size(1), tiled_mma.get_tile_size(2)),
    )

    thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tidx)
    thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tidx)
    
    tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
    tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)   

    tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
    tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

    for k in cutlass.range_dynamic(tCrA.shape[2]):
        cute.copy(tiled_copy_s2r_A, tCsA_copy_view[None, None, k], tCrA_copy_view[None, None, k])
        cute.copy(tiled_copy_s2r_B, tCsB_copy_view[None, None, k], tCrB_copy_view[None, None, k])
        cute.gemm(tiled_mma, tCrC, tCrA[None, None, k], tCrB[None, None, k], tCrC)


@cute.kernel
def gemm_f16f16f16_nn_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    sA_layout: cute.ComposedLayout,
    sB_layout: cute.ComposedLayout,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    tiler_coord = (bidx, bidy, None)
    
    smem = cutlass.utils.SmemAllocator()

    smem_storage = smem.allocate_tensor(cutlass.Uint8, 32768, 128) # 1024 align

    gA = cute.local_tile(mA[None,None,bidz], tiler=cta_tiler, coord=tiler_coord, proj=(1, None, 1))
    gB = cute.local_tile(mB[None,None,bidz], tiler=cta_tiler, coord=tiler_coord, proj=(None, 1, 1))
    gC = cute.local_tile(mC[None,None,bidz], tiler=cta_tiler, coord=tiler_coord, proj=(1, 1, None))

    sA_storage = cute.make_tensor(cute.recast_ptr(smem_storage.iterator, dtype = cutlass.Float16), sA_layout)
    sB_storage = cute.make_tensor(cute.recast_ptr(smem_storage.iterator, dtype = cutlass.Float16)+8192, sB_layout)
    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)

    tAgA = thr_copy_A.partition_S(gA)
    tBgB = thr_copy_B.partition_S(gB)
    tAsA = thr_copy_A.partition_D(sA_storage)
    tBsB = thr_copy_B.partition_D(sB_storage)    

    C_local = cute.make_fragment(128, cutlass.Float16)

    C_local.fill(0)

    cute.copy(tiled_copy_A, tAgA[None, None, None, 0], tAsA[None, None, None, 0])
    cute.copy(tiled_copy_B, tBgB[None, None, None, 0], tBsB[None, None, None, 0])

    cute.arch.cp_async_commit_group()

    for k in cutlass.range_dynamic(23): 
        cute.arch.sync_threads()

        cute.copy(tiled_copy_A, tAgA[None, None, None, k+1], tAsA[None, None, None, (k+1) & 1])
        cute.copy(tiled_copy_B, tBgB[None, None, None, k+1], tBsB[None, None, None, (k+1) & 1])

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(1)
        cute.arch.sync_threads()

        sA = sA_storage[None, None, (k & 1)]
        sB = sB_storage[None, None, (k & 1)]
        gemm_ss(sA, sB, C_local, tiled_mma)

    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    sA = sA_storage[None, None, 1]
    sB = sB_storage[None, None, 1]
    gemm_ss(sA, sB, C_local, tiled_mma)

    tCrC = cute.make_tensor(C_local.iterator, tiled_mma.partition_shape_C((cta_tiler[0], cta_tiler[1])))
    thr_mma = tiled_mma.get_slice(tidx)
    tCgC = thr_mma.partition_C(gC)
    cute.autovec_copy(tCrC, tCgC)

@cute.jit
def gemm_f16f16f16_nn(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor
):

    cta_tile_m, cta_tile_n, cta_tile_k = cta_tiler    
    stage_num = 2

    sA_layout = _make_smem_layout_AB(cutlass.Float16, a_major_mode, 128, (cta_tile_m, cta_tile_k, stage_num))
    sB_layout = _make_smem_layout_AB(cutlass.Float16, b_major_mode, 128, (cta_tile_n, cta_tile_k, stage_num))

    atom_async_copy = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(
            cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
        ),
        mA.element_type,
        num_bits_per_copy=128,
    )
    tiled_copy_A = _make_gmem_tiled_copy_AB(atom_async_copy, mA.element_type, a_major_mode, 128)
    tiled_copy_B = _make_gmem_tiled_copy_AB(atom_async_copy, mB.element_type, b_major_mode, 128)

    num_warp = (2, 2, 1)
    inst_shape = (16, 8, 16)
    op = cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float16, inst_shape )
    tC = cute.make_layout(num_warp)
    permutation_mnk = (
        num_warp[0] * inst_shape[0],
        num_warp[1] * inst_shape[1] * 2,
        num_warp[2] * inst_shape[2],
    )
    tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)

    # kernel = tensorop_kernel(mA, mB, mC, sA_layout, sB_layout, tiled_copy_A, tiled_copy_B, tiled_mma)
    kernel = gemm_f16f16f16_nn_kernel(mA, mB, mC, sA_layout, sB_layout, tiled_copy_A, tiled_copy_B, tiled_mma)
    
    # Launch with grid that covers the full output matrix
    kernel.launch(grid=(4, 8, 1),   # tilelang grid is N,M,L -> X,Y,Z
                  block=(128, 1, 1),
                  smem=32768)

def create_and_permute_tensor(l, mode0, mode1, is_mode0_major, dtype):
    # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
    # else: (l, mode0, mode1) -> (mode0, mode1, l)
    shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
    permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)

    return (
        torch.empty(*shape, dtype=torch.int32)
        .random_(-2, 2)
        .to(dtype=dtype)
        .permute(permute_order)
        .cuda()
    )


# Test the kernel
L, M, N, K = 1, 512, 1024, 768
a_major,b_major,c_major = "k","n","n"
ab_dtype = cutlass.Float16
c_dtype = cutlass.Float16
acc_dtype = cutlass.Float16
atom_layout_mnk = (2, 2, 1)
torch.manual_seed(0)

# Fixed: Create tensors with correct layout assumptions
# For k-major layout: tensor shape is (M, K) or (N, K)
a = create_and_permute_tensor(L, M, K, a_major == "m", cutlass_torch.dtype(ab_dtype))
b = create_and_permute_tensor(L, N, K, b_major == "n", cutlass_torch.dtype(ab_dtype))
c = create_and_permute_tensor(L, M, N, c_major == "m", cutlass_torch.dtype(c_dtype))


# assume input is 16B aligned
mA = (
    from_dlpack(a, assumed_align=16)
    .mark_layout_dynamic(leading_dim=(1 if a_major == "k" else 0))
    .mark_compact_shape_dynamic(
        mode=(1 if a_major == "k" else 0),
        stride_order=(2, 0, 1) if a_major == "k" else (2, 1, 0),
        divisibility=(128 // ab_dtype.width),
    )
)
mB = (
    from_dlpack(b, assumed_align=16)
    .mark_layout_dynamic(leading_dim=(1 if b_major == "k" else 0))
    .mark_compact_shape_dynamic(
        mode=(1 if b_major == "k" else 0),
        stride_order=(2, 0, 1) if b_major == "k" else (2, 1, 0),
        divisibility=(128 // ab_dtype.width),
    )
)
mC = (
    from_dlpack(c, assumed_align=16)
    .mark_layout_dynamic(leading_dim=(1 if c_major == "n" else 0))
    .mark_compact_shape_dynamic(
        mode=(1 if c_major == "n" else 0),
        stride_order=(2, 0, 1) if c_major == "n" else (2, 1, 0),
        divisibility=(128 // c_dtype.width),
    )
)


# == benchmark ==
import time
import cuda.bindings.driver as cuda

print("Compiling kernel with cute.compile ...")
start_time = time.time()
gemm_f16f16f16_nn_ = cute.compile(gemm_f16f16f16_nn, mA, mB, mC)

compilation_time = time.time() - start_time
print(f"Compilation time: {compilation_time:.4f} seconds")

gemm_f16f16f16_nn_(mA, mB, mC)
# Verify correctness - compare with torch reference
c_ref = torch.einsum("mkl,nkl->mnl", a, b).to(cutlass_torch.dtype(c_dtype))

print("max diff", torch.max(torch.abs(c.cpu() - c_ref.cpu())))
torch.testing.assert_close(c.cpu(), c_ref.cpu(), atol=1e-3, rtol=1e-3)
print("FP16 GEMM kernel test passed!")

print("Executing GEMM kernel...")
# Get current CUDA stream from PyTorch
torch_stream = torch.cuda.current_stream()

# Get the raw stream pointer as a CUstream
current_stream = cuda.CUstream(torch_stream.cuda_stream)

# Create CUDA events for timing
start_event = cuda.cuEventCreate(cuda.CUevent_flags.CU_EVENT_DEFAULT)[1]
end_event = cuda.cuEventCreate(cuda.CUevent_flags.CU_EVENT_DEFAULT)[1]

warmup_iterations = 10
iterations = 100
# Warmup
for _ in range(warmup_iterations):
    gemm_f16f16f16_nn_(mA, mB, mC)

# Use the current stream for CUDA events instead of the default stream
# Record start event
cuda.cuEventRecord(start_event, current_stream)

# Execute the kernel
for _ in range(iterations):
    gemm_f16f16f16_nn_(mA, mB, mC)

# Record end event
cuda.cuEventRecord(end_event, current_stream)
cuda.cuEventSynchronize(end_event)

# Calculate elapsed time
err, elapsed_time = cuda.cuEventElapsedTime(start_event, end_event)

# Print execution results
print(f"Kernel execution time: {elapsed_time / iterations:.4f} ms")

# Destroy events
cuda.cuEventDestroy(start_event)
cuda.cuEventDestroy(end_event)
    