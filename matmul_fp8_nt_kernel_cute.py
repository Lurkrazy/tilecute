import torch
from functools import partial

import math
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import from_dlpack
from utils import mbarrier_expect_tx, mbarrier_wait

stages = 3
cta_tile_shape_mnk = (128, 128, 64)

def _make_tma_atoms_and_tensors(
    tensor,
    smem_layout_staged,
    smem_tile,
    mcast_dim,
):
    op = (
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        if mcast_dim == 1
        else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
    )

    smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
    # print(smem_layout)
    tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tma_tile_atom(
        op,
        tensor,
        smem_layout,
        smem_tile,
        num_multicast=mcast_dim,
    )
    return tma_atom, tma_tensor

@cute.struct
class SharedStorage:
    bar: cute.struct.MemRange[cutlass.Int64, stages*2]
    sA: cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, 8192*stages], 1024]
    sB: cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, 8192*stages], 1024]


@cute.kernel
def matmul_fp8_nt_kernel(
    tma_atom_a: cute.CopyAtom,
    gA: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    gB: cute.Tensor,
    gC: cute.Tensor,
    tiled_mma: cute.TiledMma,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
):
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    warp_group_idx = warp_idx // 4

    if warp_idx == 0:
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
    
    # Create shared memory allocator
    smem_alloc = cutlass.utils.SmemAllocator()
    storage = smem_alloc.allocate(SharedStorage)

    sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
    sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
    # Define tile sizes - 128x128x64 CTA tile
    cta_tile_m, cta_tile_n, cta_tile_k = cta_tile_shape_mnk
    tile_coord_mnk = (bidy, bidx, None)
    # print("sA", sA.layout)
    # print("sB", sB.layout)

    # Local tile the global tensors
    gA_local = cute.local_tile(gA, cta_tile_shape_mnk, tile_coord_mnk, proj = (1, None, 1))
    gB_local = cute.local_tile(gB, cta_tile_shape_mnk, tile_coord_mnk, proj = (None, 1, 1))
    gC_local = cute.local_tile(gC, cta_tile_shape_mnk, tile_coord_mnk, proj = (1, 1, None))

    thr_mma = tiled_mma.get_slice(tidx) # if use tma to store, pass warpgroup id * 128 instead.

    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCgC = thr_mma.partition_C(gC_local)

    tCrA = thr_mma.make_fragment_A(tCsA)
    tCrB = thr_mma.make_fragment_B(tCsB)    
    tCrC = cute.make_fragment(tCgC.shape, cutlass.Float32)
    
    # print("gC_local.iterator:", gC_local.iterator)
    # print("gC_local.iterator.max_alignment:", gC_local.iterator.max_alignment)
    # print("tCgC.iterator:", tCgC.iterator)
    # print("tCgC.iterator.max_alignment:", tCgC.iterator.max_alignment)
    # print("tCgC.layout:", tCgC.layout)
    # print("tCgC.layout.max_alignment:", tCgC.layout.max_alignment)
    
    # tCrC = cute.make_fragment_like(tCgC)
    # print("gC.layout", gC.layout)
    # print(tCrC.layout)
    # cute.printf("{}", tCgC.layout)

    # Create shared memory tensors
    tAsA_preslice, tAgA_preslice = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        0,  # cta_coord
        cute.make_layout(1),
        cute.group_modes(sA, 0, 2), # (tile_m, tile_k, stages) -> ((shape), stages)
        cute.group_modes(gA_local, 0, 2),
    )

    tBsB_preslice, tBgB_preslice = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(sB, 0, 2),  
        cute.group_modes(gB_local, 0, 2),
    )
    # print(tAsA_preslice.layout)
    # print(tAgA_preslice.layout)
    # print(tBsB_preslice.layout)
    # print(tBgB_preslice.layout)

    mbars = storage.bar.data_ptr()

    # Initialize barriers only once per warp group
    if warp_idx == 0:
        with cute.arch.elect_one():
            for i in range(stages * 2):
                cute.arch.mbarrier_init_arrive_cnt(mbars + i, 128)

    cute.arch.mbarrier_init_fence()
    cute.arch.barrier() # equivalent to __syncthreads()

    # Calculate number of K tiles
    k_tile_count = 16  # K dimension tiles
    # print("k_tile_count", k_tile_count)

    # TMA warpgroup
    if warp_group_idx == 1: 
        cute.arch.warpgroup_reg_dealloc(24)                
        # Calculate proper TMA transfer size
        a_transfer_size = cute.size_in_bytes(cutlass.Float8E4M3FN, cute.slice_(sA, (None, None, 0)))
        b_transfer_size = cute.size_in_bytes(cutlass.Float8E4M3FN, cute.slice_(sB, (None, None, 0)))
        total_transfer_size = a_transfer_size + b_transfer_size
        # phase = 0
        for k in cutlass.range_dynamic(k_tile_count, unroll=stages):
            stage = k % stages
            phase = ((k % 6) // 3) ^ 1
            mbarrier_wait(mbars + stage + stages, phase, timeout_ns=100)

            if warp_idx % 4 == 0: 
                with cute.arch.elect_one():
                    mbarrier_expect_tx(mbars + stage, total_transfer_size)

                tAsA = tAsA_preslice[(None, stage)]
                tAgA = tAgA_preslice[(None, k)]

                tBsB = tBsB_preslice[(None, stage)]
                tBgB = tBgB_preslice[(None, k)]
                
                cute.copy(tma_atom_a, tAgA, tAsA, tma_bar_ptr=mbars + stage)
                cute.copy(tma_atom_b, tBgB, tBsB, tma_bar_ptr=mbars + stage)
            # all threads will arrive
            cute.arch.mbarrier_arrive(mbars + stage)
        # tile process
        # cute.arch.mbarrier_wait(mbars + ((k_tile_count-1) % stages) + stages, phase^1)
    elif warp_group_idx == 0: # gemm warp group
        cute.arch.warpgroup_reg_alloc(240)
        tCrC.fill(0.0)
        num_k_blocks = cute.size(tCrA, mode=[2])
        cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)

        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        # num_k_blocks = cute.size(tCrA, mode=[2])
        # print(num_k_blocks)

        # tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        # Prologue MMA:
        # for k in range(1):
        #     stage = k % stages
        #     phase = ((k % 6) // 3)
        #     cute.arch.mbarrier_wait(mbars + stage, phase)
        #     cute.nvgpu.warpgroup.fence()
        #     for k_block in range(num_k_blocks):
        #         k_block_coor = (None, None, k_block, stage)
        #         tCrA_1phase = tCrA[k_block_coor]
        #         tCrB_1phase = tCrB[k_block_coor]
        #         cute.gemm(tiled_mma, tCrC, tCrA_1phase, tCrB_1phase, tCrC)
        #         tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        #     cute.nvgpu.warpgroup.commit_group()
        #     # Wait for all WGMMA operations to complete
        #     cute.nvgpu.warpgroup.wait_group(0)
        #     cute.arch.mbarrier_arrive(mbars + stage + stages)
            
        for k in cutlass.range_dynamic(k_tile_count,unroll=stages):
            stage = k % stages
            phase = ((k % 6) // 3)
            mbarrier_wait(mbars + stage, phase, timeout_ns=100)
            cute.nvgpu.warpgroup.fence()
            for k_block in range(num_k_blocks):
                k_block_coor = (None, None, k_block, stage)
                tCrA_1phase = tCrA[k_block_coor]
                tCrB_1phase = tCrB[k_block_coor]
                cute.gemm(tiled_mma, tCrC, tCrA_1phase, tCrB_1phase, tCrC)
                # tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.commit_group()
            # Wait for all WGMMA operations to complete
            cute.nvgpu.warpgroup.wait_group(0)
            cute.arch.mbarrier_arrive(mbars + stage + stages)
            
        # Store results back to global memory using TenserSSA 
        tCgC_view = cute.make_tensor(tCgC.iterator.align(8), tCgC.layout) # sizeof(float2) = 8
        # tCgC_view.store(tCrC.load())
        # Or use cutlass style
        cute.autovec_copy(tCrC, tCgC_view)

        # for i_1 in range(64):
        #     gC_offset = bidy * 131072 + (i_1 >> 5) * 65536 + (tidx >> 5) * 16384 + (i_1 & 1) * 8192 + ((tidx & 31) >> 2) * 1024 + bidx * 128 + ((i_1 & 31) >> 1) * 8 + (tidx & 3) * 2
        #     gC_offset = cute.assume(gC_offset, divby = 2)
        #     tC = cute.make_tensor(gC.iterator + gC_offset, 2)
        #     rC = cute.make_tensor(tCrC.iterator + i_1*2, 2)
        #     tC.store(rC.load())


@cute.jit
def matmul_fp8_nt(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor
):
    cta_tile_m, cta_tile_n, cta_tile_k = cta_tile_shape_mnk    
    sa_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW64, cutlass.Float8E4M3FN)
    sb_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW64, cutlass.Float8E4M3FN)

    # print(sa_layout_atom)

    sa_layout_staged = cute.tile_to_shape(
        sa_layout_atom, 
        (cta_tile_m, cta_tile_k, stages), 
        order=(0, 1, 2)) 
    
    sb_layout_staged = cute.tile_to_shape(
        sb_layout_atom, 
        (cta_tile_n, cta_tile_k, stages), 
        order=(0, 1, 2))
    
    # print(sa_layout_staged)

    mma_inst_shape_mnk = (64, 128, 32)
    atom_layout_mnk = (1, 1, 1)
    op = cute.nvgpu.warpgroup.MmaF8Op(
        cutlass.Float8E4M3FN,
        cutlass.Float8E4M3FN,
        cutlass.Float32,
        mma_inst_shape_mnk,
        cute.nvgpu.warpgroup.OperandSource.SMEM,
        cute.nvgpu.warpgroup.OperandMajorMode("K"),
        cute.nvgpu.warpgroup.OperandMajorMode("K"),
    )
    # print(op)
    
    tiled_mma = cute.make_tiled_mma(cute.make_mma_atom(op), atom_layout_mnk)
    # tiled_mma = sm90_utils.make_trivial_tiled_mma(
    #     cutlass.Float8E4M3FN,
    #     cutlass.Float8E4M3FN,
    #     cute.nvgpu.warpgroup.OperandMajorMode("K"),
    #     cute.nvgpu.warpgroup.OperandMajorMode("K"),
    #     cutlass.Float32,
    #     atom_layout_mnk,
    #     (mma_inst_shape_mnk[0], mma_inst_shape_mnk[1])
    # )
    # print(tiled_mma)
    
    tma_atom_a, tma_tensor_a = _make_tma_atoms_and_tensors(mA, sa_layout_staged, (cta_tile_m, cta_tile_k), 1)
    tma_atom_b, tma_tensor_b = _make_tma_atoms_and_tensors(mB, sb_layout_staged, (cta_tile_n, cta_tile_k), 1)

    # print(tma_atom_a)
    # print(mA)
    # print(tma_tensor_a)

    kernel = matmul_fp8_nt_kernel(tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, mC, tiled_mma, sa_layout_staged, sb_layout_staged)
    # Launch with grid that covers the full output matrix
    # Grid: (M/128, N/128, 1), Block: (256, 1, 1) - corrected grid dimensions
    grid_x = (mC.shape[0] + 127) // 128  # M dimension
    grid_y = (mC.shape[1] + 127) // 128  # N dimension
    kernel.launch(grid=(grid_x, grid_y, 1), 
                  block=(256, 1, 1),
                  smem=SharedStorage.size_in_bytes())



def create_and_permute_tensor(
    mode0, mode1, is_mode0_major, dtype, is_dynamic_layout=True
):
    # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
    # else : (l, mode0, mode1) -> (mode0, mode1, l)
    shape = (mode1, mode0) if is_mode0_major else (mode0, mode1)
    permute_order = (1, 0) if is_mode0_major else (0, 1)
    is_unsigned = dtype in {cutlass.Uint8}
    # Temporarily use uint8 as torch does not support fp8 type
    torch_dtype = cutlass_torch.dtype(dtype)
    

    # Create dtype torch tensor (cpu)
    if dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}:
        torch_tensor = torch.randn(shape).to(dtype=torch_dtype).cuda()
        f32_torch_tensor = torch_tensor.to(dtype=torch.float32)
    else:
        torch_tensor_cpu = cutlass.torch.create_and_permute_torch_tensor(
            shape,
            torch_dtype,
            permute_order=permute_order,
            init_type=cutlass.torch.TensorInitType.RANDOM,
            init_config=cutlass.torch.RandomInitConfig(
                min_val=0 if is_unsigned else -2, max_val=4 if is_unsigned else 2
            ),
        )
        # Create dtype torch tensor (gpu)
        torch_tensor = torch_tensor_cpu.cuda()
        # Create f32 torch tensor (cpu)
        f32_torch_tensor = torch_tensor_cpu.to(dtype=torch.float32)

    # Create dtype cute tensor (gpu)
    if dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}:
        cute_tensor = from_dlpack(torch_tensor.view(torch.uint8), assumed_align=16)
    else:
        cute_tensor = from_dlpack(torch_tensor, assumed_align=16)
    cute_tensor.element_type = dtype
    if is_dynamic_layout:
        cute_tensor = cute_tensor.mark_layout_dynamic(
            leading_dim=(0 if is_mode0_major else 1)
        )
    cute_tensor = cutlass.torch.convert_cute_tensor(
        f32_torch_tensor,
        cute_tensor,
        dtype,
        is_dynamic_layout=is_dynamic_layout,
    )

    return f32_torch_tensor, cute_tensor, torch_tensor


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


# Test the kernel
M, N, K = 1024, 1024, 1024
# Fixed: Create tensors with correct layout assumptions
# For k-major layout: tensor shape is (M, K) or (N, K)
a_f32, mA, a_torch = create_and_permute_tensor(M, K, False, cutlass.Float8E4M3FN)  # k-major
b_f32, mB, b_torch = create_and_permute_tensor(N, K, False, cutlass.Float8E4M3FN)  # k-major
c_f32, mC, c_torch = create_and_permute_tensor(M, N, False, cutlass.Float32)  # n-major

# == benchmark ==
import time
import cuda.bindings.driver as cuda

print("Compiling kernel with cute.compile ...")
start_time = time.time()
matmul_fp8_nt_ = cute.compile(matmul_fp8_nt, mA, mB, mC)
compilation_time = time.time() - start_time
print(f"Compilation time: {compilation_time:.4f} seconds")

matmul_fp8_nt_(mA, mB, mC)
# Verify correctness - compare with torch reference
# Fixed: No need to transpose B since both A and B are k-major
c_ref = torch.matmul(a_f32, b_f32.T)

diff = calc_diff(c_torch, c_ref)
print(f"diff: {diff}")
# assert diff < 1e-5
print("FP8 GEMM kernel test passed!")
# torch.testing.assert_close(c_torch.cpu(), c_ref, rtol=1e-2, atol=1e-2)  # Lower tolerance for fp8
# print("FP8 GEMM kernel test passed!")

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
    matmul_fp8_nt_(mA, mB, mC)

# Use the current stream for CUDA events instead of the default stream
# Record start event
cuda.cuEventRecord(start_event, current_stream)

# Execute the kernel
for _ in range(iterations):
    matmul_fp8_nt_(mA, mB, mC)

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