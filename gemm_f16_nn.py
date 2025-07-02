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
def gemm_ss(  # 128,128,32 gemm
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
        cute.nvgpu.warp.LdMatrix8x8x16bOp(
            a_major_mode != cutlass.utils.LayoutEnum.ROW_MAJOR, 4
        ),
        sA.element_type,
    )
    atom_copy_s2r_B = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(
            b_major_mode != cutlass.utils.LayoutEnum.ROW_MAJOR, 4
        ),
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

    print(sA_layout)
    print(sB_layout)

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

    # cute.copy(tiled_copy_A, tAgA[None, None, None, 0], tAsA[None, None, None, 0])
    # cute.copy(tiled_copy_B, tBgB[None, None, None, 0], tBsB[None, None, None, 0])

    bidx, bidy = bidy, bidx

    for i_1 in range(4):

        smem_offset = ((((i_1 * 2048) + (((tidx) >> 2) * 64)) + ((((((tidx) & 31) >> 4) + (((tidx) & 3) >> 1)) & 1) * 32)) + ((((((tidx) & 15) >> 3) + ((tidx) & 1)) & 1) * 16))
        
        global_offset = (((((bidy) * 98304) + (i_1 * 24576)) + (((tidx) >> 2) * 768)) + (((tidx) & 3) * 8))

        cp_async_shared_global(
            dst = smem_storage.iterator + smem_offset,
            src = mA.iterator + global_offset, 
            cp_size = 16,
            modifier = nvvm.LoadCacheModifierKind.CG  # enable L2 prefetch
        )

    for i_2 in range(4):
        
        smem_offset = ((((((((((tidx) & 15) >> 3) * 4096) + (i_2 * 1024)) + (((tidx) >> 4) * 128)) + (((((tidx) >> 6) + (((tidx) & 7) >> 2)) & 1) * 64)) + ((((((tidx) & 63) >> 5) + (((tidx) & 3) >> 1)) & 1) * 32)) + ((((((tidx) & 31) >> 4) + ((tidx) & 1)) & 1) * 16)) + 16384)
        
        global_offset = ((((i_2 * 8192) + (((tidx) >> 4) * 1024)) + ((bidx) * 128)) + (((tidx) & 15) * 8))

        cp_async_shared_global(
            dst = smem_storage.iterator + smem_offset,
            src = mB.iterator + global_offset, 
            cp_size = 16,
            modifier = nvvm.LoadCacheModifierKind.CG  # enable L2 prefetch
        )


    cute.arch.cp_async_commit_group()

    for k in cutlass.range_dynamic(23): 
        cute.arch.sync_threads()

        # cute.copy(tiled_copy_A, tAgA[None, None, None, k+1], tAsA[None, None, None, (k+1) & 1])
        # cute.copy(tiled_copy_B, tBgB[None, None, None, k+1], tBsB[None, None, None, (k+1) & 1])

        for i_3 in range(4):

            smem_offset = (((((((k + 1) & 1) * 8192) + (i_3 * 2048)) + (((tidx) >> 2) * 64)) + ((((((tidx) & 31) >> 4) + (((tidx) & 3) >> 1)) & 1) * 32)) + ((((((tidx) & 15) >> 3) + ((tidx) & 1)) & 1) * 16))

            global_offset = (((((((bidy) * 98304) + (i_3 * 24576)) + (((tidx) >> 2) * 768)) + (k * 32)) + (((tidx) & 3) * 8)) + 32)

            cp_async_shared_global(
                dst = smem_storage.iterator + smem_offset,
                src = mA.iterator + global_offset, 
                cp_size = 16,
                modifier = nvvm.LoadCacheModifierKind.CG  # enable L2 prefetch
            )
        
        for i_4 in range(4):

            smem_offset = ((((((((((k + 1) & 1) * 8192) + ((((tidx) & 15) >> 3) * 4096)) + (i_4 * 1024)) + (((tidx) >> 4) * 128)) + (((((tidx) >> 6) + (((tidx) & 7) >> 2)) & 1) * 64)) + ((((((tidx) & 63) >> 5) + (((tidx) & 3) >> 1)) & 1) * 32)) + ((((((tidx) & 31) >> 4) + ((tidx) & 1)) & 1) * 16)) + 16384)

            global_offset = ((((((k * 32768) + (i_4 * 8192)) + (((tidx) >> 4) * 1024)) + ((bidx) * 128)) + (((tidx) & 15) * 8)) + 32768)

            cp_async_shared_global(
                dst = smem_storage.iterator + smem_offset,
                src = mB.iterator + global_offset, 
                cp_size = 16,
                modifier = nvvm.LoadCacheModifierKind.CG  # enable L2 prefetch
            )

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(1)
        cute.arch.sync_threads()

        sA = cute.make_tensor(cute.recast_ptr(smem_storage.iterator,dtype = cutlass.Float16)+((k & 1) * 4096), sA_layout)[None, None, 0]
        sB = cute.make_tensor(cute.recast_ptr(smem_storage.iterator,dtype = cutlass.Float16)+(((k & 1) * 4096) + 8192), sB_layout)[None, None, 0]
        # sA = sA_storage[None, None, (k & 1)]
        # sB = sB_storage[None, None, (k & 1)]
        gemm_ss(sA, sB, C_local, tiled_mma)

    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    sA = cute.make_tensor(cute.recast_ptr(smem_storage.iterator, dtype = cutlass.Float16) + 4096, sA_layout)[None, None, 0]
    sB = cute.make_tensor(cute.recast_ptr(smem_storage.iterator, dtype = cutlass.Float16) + 12288, sB_layout)[None, None, 0]
    # sA = sA_storage[None, None, 1]
    # sB = sB_storage[None, None, 1]
    gemm_ss(sA, sB, C_local, tiled_mma)

    # tCrC = cute.make_tensor(C_local.iterator, tiled_mma.partition_shape_C((cta_tiler[0], cta_tiler[1])))
    # print("tCrC layout", tCrC.layout)
    # thr_mma = tiled_mma.get_slice(tidx)
    # tCgC = thr_mma.partition_C(gC)
    # cute.autovec_copy(tCrC, tCgC)
    # bidx, bidy = bidy, bidx
    for i_5 in range(64):
        global_offset = ((((((((((bidy) * 131072) + (((i_5 & 7) >> 1) * 32768)) + ((((tidx) & 63) >> 5) * 16384)) + ((i_5 & 1) * 8192)) + ((((tidx) & 31) >> 2) * 1024)) + ((bidx) * 128)) + ((i_5 >> 3) * 16)) + (((tidx) >> 6) * 8)) + (((tidx) & 3) * 2))
        global_offset = cute.assume(global_offset, divby=2)
        tC = cute.make_tensor(mC.iterator + global_offset, 2)
        rC = cute.make_tensor(C_local.iterator + i_5 * 2, 2)
        tC.store(rC.load())


@cute.kernel
def gemm_f16f16f16_nn_kernel_test(
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
    bidy, bidx, bidz = cute.arch.block_idx()
    bdimx, bdimy, bdimz = cute.arch.block_dim()

    # bidy, bidx = bidx, bidy # tilelang
    
    smem = cutlass.utils.SmemAllocator()

    smem_storage = smem.allocate_tensor(cutlass.Uint8, 32768, 128) # 1024 align

    C_local = cute.make_fragment(128, cutlass.Float16)

    C_local.fill(0)

    for i_1 in range(4):

        smem_offset = ((((i_1 * 2048) + (((tidx) >> 2) * 64)) + ((((((tidx) & 31) >> 4) + (((tidx) & 3) >> 1)) & 1) * 32)) + ((((((tidx) & 15) >> 3) + ((tidx) & 1)) & 1) * 16))
        
        global_offset = (((((bidy) * 98304) + (i_1 * 24576)) + (((tidx) >> 2) * 768)) + (((tidx) & 3) * 8))

        cp_async_shared_global(
            dst = smem_storage.iterator + smem_offset,
            src = mA.iterator + global_offset, 
            cp_size = 16,
            modifier = nvvm.LoadCacheModifierKind.CG  # enable L2 prefetch
        )

    for i_2 in range(4):
        
        smem_offset = ((((((((((tidx) & 15) >> 3) * 4096) + (i_2 * 1024)) + (((tidx) >> 4) * 128)) + (((((tidx) >> 6) + (((tidx) & 7) >> 2)) & 1) * 64)) + ((((((tidx) & 63) >> 5) + (((tidx) & 3) >> 1)) & 1) * 32)) + ((((((tidx) & 31) >> 4) + ((tidx) & 1)) & 1) * 16)) + 16384)
        
        global_offset = ((((i_2 * 8192) + (((tidx) >> 4) * 1024)) + ((bidx) * 128)) + (((tidx) & 15) * 8))

        cp_async_shared_global(
            dst = smem_storage.iterator + smem_offset,
            src = mB.iterator + global_offset, 
            cp_size = 16,
            modifier = nvvm.LoadCacheModifierKind.CG  # enable L2 prefetch
        )

    cute.arch.cp_async_commit_group()

    for k in cutlass.range_dynamic(23): 
        cute.arch.sync_threads()

        for i_3 in range(4):

            smem_offset = (((((((k + 1) & 1) * 8192) + (i_3 * 2048)) + (((tidx) >> 2) * 64)) + ((((((tidx) & 31) >> 4) + (((tidx) & 3) >> 1)) & 1) * 32)) + ((((((tidx) & 15) >> 3) + ((tidx) & 1)) & 1) * 16))

            global_offset = (((((((bidy) * 98304) + (i_3 * 24576)) + (((tidx) >> 2) * 768)) + (k * 32)) + (((tidx) & 3) * 8)) + 32)

            cp_async_shared_global(
                dst = smem_storage.iterator + smem_offset,
                src = mA.iterator + global_offset, 
                cp_size = 16,
                modifier = nvvm.LoadCacheModifierKind.CG  # enable L2 prefetch
            )

        for i_4 in range(4):

            smem_offset = ((((((((((k + 1) & 1) * 8192) + ((((tidx) & 15) >> 3) * 4096)) + (i_4 * 1024)) + (((tidx) >> 4) * 128)) + (((((tidx) >> 6) + (((tidx) & 7) >> 2)) & 1) * 64)) + ((((((tidx) & 63) >> 5) + (((tidx) & 3) >> 1)) & 1) * 32)) + ((((((tidx) & 31) >> 4) + ((tidx) & 1)) & 1) * 16)) + 16384)

            global_offset = ((((((k * 32768) + (i_4 * 8192)) + (((tidx) >> 4) * 1024)) + ((bidx) * 128)) + (((tidx) & 15) * 8)) + 32768)

            cp_async_shared_global(
                dst = smem_storage.iterator + smem_offset,
                src = mB.iterator + global_offset, 
                cp_size = 16,
                modifier = nvvm.LoadCacheModifierKind.CG  # enable L2 prefetch
            )

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(1)
        cute.arch.sync_threads()

        sA = cute.make_tensor(cute.recast_ptr(smem_storage.iterator,dtype = cutlass.Float16)+((k & 1) * 4096), sA_layout)[None, None, 0]
        sB = cute.make_tensor(cute.recast_ptr(smem_storage.iterator,dtype = cutlass.Float16)+(((k & 1) * 4096) + 8192), sB_layout)[None, None, 0]
        gemm_ss(sA, sB, C_local, tiled_mma)

    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()
    sA = cute.make_tensor(cute.recast_ptr(smem_storage.iterator, dtype = cutlass.Float16) + 4096, sA_layout)[None, None, 0]
    sB = cute.make_tensor(cute.recast_ptr(smem_storage.iterator, dtype = cutlass.Float16) + 12288, sB_layout)[None, None, 0]
    gemm_ss(sA, sB, C_local, tiled_mma)

    for i_5 in range(64):
        global_offset = ((((((((((bidy) * 131072) + (((i_5 & 7) >> 1) * 32768)) + ((((tidx) & 63) >> 5) * 16384)) + ((i_5 & 1) * 8192)) + ((((tidx) & 31) >> 2) * 1024)) + ((bidx) * 128)) + ((i_5 >> 3) * 16)) + (((tidx) >> 6) * 8)) + (((tidx) & 3) * 2))
        global_offset = cute.assume(global_offset, divby=2)
        tC = cute.make_tensor(mC.iterator + global_offset, 2)
        rC = cute.make_tensor(C_local.iterator + i_5 * 2, 2)
        tC.store(rC.load())

@cute.kernel
def tensorop_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    sA_layout: cute.ComposedLayout,
    sB_layout: cute.ComposedLayout,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
):
    # Thread index, block index
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    tiler_coord = (bidx, bidy, None)

    # ///////////////////////////////////////////////////////////////////////////////
    # Get the appropriate tiles for this thread block.
    # gA: (BLK_M, BLK_N, k), gB: (BLK_N, BLK_K, k), gC: (BLK_M, BLK_N)
    # ///////////////////////////////////////////////////////////////////////////////
    gA = cute.local_tile(
        mA[None, None, bidz],
        tiler=cta_tiler,
        coord=tiler_coord,
        proj=(1, None, 1),
    )
    gB = cute.local_tile(
        mB[None, None, bidz],
        tiler=cta_tiler,
        coord=tiler_coord,
        proj=(None, 1, 1),
    )
    gC = cute.local_tile(
        mC[None, None, bidz],
        tiler=cta_tiler,
        coord=tiler_coord,
        proj=(1, 1, None),
    )
    print("mA layout", mA.layout)
    print("mB layout", mB.layout)
    print("gA layout", gA.layout)
    print("gB layout", gB.layout)

    # By default, if the tensor k mode does not divide into the tile k
    # size, then last tiles in the k dimension are irregular.
    # Instead, make the first tiles irregular when k is irregular.
    # This allows us to handle the irregular tile first to avoid
    # checking for this condition within the mainloop.

    # input is 16B aligned
    gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
    gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

    # ///////////////////////////////////////////////////////////////////////////////
    # Create shared memory buffers and get the appropriate fragments for this thread.
    # sA:   (BLK_M, BLK_K, PIPE)       , sB:   (BLK_N, BLK_K, PIPE)
    # tAgA: (CPY, CPY_M, CPY_K, k)     , tBgB: (CPY, CPY_N, CPY_K, k)
    # tAsA: (CPY, CPY_M, CPY_K, PIPE)  , tBsB: (CPY, CPY_N, CPY_K, PIPE)
    # ///////////////////////////////////////////////////////////////////////////////
    # Shared memory buffer
    smem = cutlass.utils.SmemAllocator()

    sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
    sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)

    tAgA = thr_copy_A.partition_S(gA)
    tAsA = thr_copy_A.partition_D(sA)
    tBgB = thr_copy_B.partition_S(gB)
    tBsB = thr_copy_B.partition_D(sB)


    # ///////////////////////////////////////////////////////////////////////////////
    # Prefetch Prologue
    # ///////////////////////////////////////////////////////////////////////////////
    # Clear the smem tiles to account for predicated off loads
    # tAsA.fill(0)
    # tBsB.fill(0)
    cute.arch.sync_threads()
    # Start async loads for the first k-tile. Here we take care of the k residue
    # via if/else check along the k dimension. Because we shifted the identity tensor
    # by the residue_k and because the identity tensor is a counting tensor, the
    # values of any identity tensor element that is poison is less than -1
    num_smem_stages = cute.size(tAsA, mode=[3])
    print("num_smem_stages", num_smem_stages)
    k_tile_count = cute.size(tAgA, mode=[3])

    k_tile_index = cutlass.Int32(0)

    cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index], tAsA[None, None, None, 0])
    cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index], tBsB[None, None, None, 0])

    k_tile_index = k_tile_index + 1
    cute.arch.cp_async_commit_group()

    # ///////////////////////////////////////////////////////////////////////////////
    # Tile MMA compute thread partitions and allocate accumulators
    # ///////////////////////////////////////////////////////////////////////////////
    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    
    print("tCsA layout", tCsA.layout)
    print("tCsB layout", tCsB.layout)

    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgC)
    # Clear the accumulator
    tCrC.fill(0.0)

    # ///////////////////////////////////////////////////////////////////////////////
    # Copy Atom A/B retiling
    # ///////////////////////////////////////////////////////////////////////////////

    # Create the copy atoms for the copy from shared memory to register
    atom_copy_s2r_A = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(
            a_major_mode != cutlass.utils.LayoutEnum.ROW_MAJOR, 4
        ),
        mA.element_type,
    )
    atom_copy_s2r_B = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(
            b_major_mode != cutlass.utils.LayoutEnum.ROW_MAJOR, 4
        ),
        mB.element_type,
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
    print("tCrA_copy_view layout",tCrA_copy_view.layout)
    print("tCrB_copy_view layout",tCrB_copy_view.layout)

    for k_tile in cutlass.range_dynamic(k_tile_count, unroll=1):
        if k_tile < k_tile_count - 1:
            cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index], tAsA[None, None, None, (k_tile+1) & 1])
            cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index], tBsB[None, None, None, (k_tile+1) & 1])
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(1)
            cute.arch.sync_threads()
        tCsA_p = tCsA_copy_view[None, None, None, (k_tile) & 1]
        tCsB_p = tCsB_copy_view[None, None, None, (k_tile) & 1]
        for k in cutlass.range_dynamic(cute.size(tCrA, mode=[2])):
            # Load A, B from shared memory to registers for k 
            cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, k], tCrA_copy_view[None, None, k])
            cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, k], tCrB_copy_view[None, None, k])
            cute.gemm(tiled_mma, tCrC, tCrA[None, None, k], tCrB[None, None, k], tCrC)
                
    
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()
    cute.autovec_copy(tCrC, tCgC)

@cute.jit
def gemm_f16f16f16_nn(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor
):

    print("mA layout", mA.layout)
    print("mB layout", mB.layout)
    print("mC layout", mC.layout)

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
    kernel.launch(grid=(4, 8, 1), 
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


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


class TensorOpGemm:
    def __init__(
        self,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        atom_layout_mnk: Tuple[int, int, int],
    ):
        self.ab_dtype = ab_dtype
        self.c_dtype = c_dtype
        self.acc_dtype = acc_dtype
        self.cta_tiler = (128, 256, 32)
        self.num_stages = 2
        self.atom_layout_mnk = atom_layout_mnk
        atom_lay_M, atom_lay_N, atom_lay_K = self.atom_layout_mnk
        self.num_threads = atom_lay_M * atom_lay_N * atom_lay_K * 32

        self.bM, self.bN, self.bK = self.cta_tiler
        self.mma_inst_shape = (16, 8, 16)
        mmaM, mmaN, mmaK = self.mma_inst_shape

        assert (
            self.bM % (atom_lay_M * mmaM) == 0
        ), "bM must be divisible by MMA instruction"
        assert (
            self.bN % (atom_lay_N * mmaN) == 0
        ), "bN must be divisible by MMA instruction"
        assert atom_lay_K == 1, "this example does not support atom layout K > 1"
        assert self.bK % mmaK == 0, "bK must be divisible by MMA instruction"
        # assert self.num_stages >= 3, "num_stages must be greater than or equal to 3"

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        # The grid divides the problems's M, N, and L dimensions by the
        # respective modes of the tile shape (bM, bN, 1). The K dimension is
        # handled within a block via a multistage process.

        self.a_major_mode = cutlass.utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = cutlass.utils.LayoutEnum.from_tensor(mB)
        self.c_major_mode = cutlass.utils.LayoutEnum.from_tensor(mC)

        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout:
        # ///////////////////////////////////////////////////////////////////////////////

        # Creates a layout with the size required for the provided tile
        # size and num stages (stages are used for K dimension) that is also
        # sectioned into 64x8 or 8x32 layout atoms. The swizzle is set so that
        # the atom for the shared memory -> register copy does not encounter
        # bank conflicts

        # assume the input is 16B align
        ab_copy_bits = 128
        sA_layout = self._make_smem_layout_AB(
            mA.element_type,
            self.a_major_mode,
            ab_copy_bits,
            (self.cta_tiler[0], self.cta_tiler[2], self.num_stages),
        )
        
        sB_layout = self._make_smem_layout_AB(
            mB.element_type,
            self.b_major_mode,
            ab_copy_bits,
            (self.cta_tiler[1], self.cta_tiler[2], self.num_stages),
        )
        print("sA_layout",sA_layout)
        print("sB_layout",sB_layout)
        # Creates a similar layout but without num_stages or layout atoms
        sC_layout = self._make_smem_layout_C(
            mC.element_type,
            self.c_major_mode,
            ab_copy_bits,
            (self.cta_tiler[0], self.cta_tiler[1]),
        )

        # Shared memory allocated for operations with A, B will be
        # overwritten for operations on C. This is to improve performance
        # by reducing the size of shared memory requested by each block
        smem_size = max(
            cute.size_in_bytes(mC.element_type, sC_layout),
            cute.size_in_bytes(mA.element_type, sA_layout)
            + cute.size_in_bytes(mB.element_type, sB_layout),
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Tiled copy:
        # The majorness of tA/tB/tC follows the majorness of gA/gB/gC,
        # enabling merged accesses to global memory for faster data
        # transfer between global and shared memory.
        # ///////////////////////////////////////////////////////////////////////////////

        # Create a copy atom for a global to shared memory asynchronous copy
        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mA.element_type,
            num_bits_per_copy=ab_copy_bits,
        )

        # Create thread layouts for tiled copy from the copy atom where the
        # thread layout simply follows the leading dimension of the tensor
        tiled_copy_A = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mA.element_type, self.a_major_mode, ab_copy_bits
        )
        tiled_copy_B = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mB.element_type, self.b_major_mode, ab_copy_bits
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Tiled MMA
        # ///////////////////////////////////////////////////////////////////////////////

        # Creates a mma atom with 16x8x16 shape for MNK
        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.ab_dtype, self.acc_dtype, self.mma_inst_shape
        )

        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            # if atom layout's N-mode is 1, to leverage the largest coalesced
            # shared memory -> register copy, set the tiled mma's N mode to 16
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )

        # Created a tiled mma that tiles the atom according to specified layout.
        # For a 2x2x1 atom layout, the mma atom is duplicated 4 times, twice
        # across M and twice across N
        tC = cute.make_layout(self.atom_layout_mnk)
        tiled_mma = cute.make_tiled_mma(
            op,
            tC,
            permutation_mnk=permutation_mnk,
        )

        # grid_dim: ((m + BLK_M - 1) // BLK_M, (n + BLK_N - 1) // BLK_N, l)
        grid_dim = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))

        self.kernel(
            mA,
            mB,
            mC,
            sA_layout,
            sB_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=smem_size,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        tiler_coord = (bidx, bidy, None)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get the appropriate tiles for this thread block.
        # gA: (BLK_M, BLK_N, k), gB: (BLK_N, BLK_K, k), gC: (BLK_M, BLK_N)
        # ///////////////////////////////////////////////////////////////////////////////
        gA = cute.local_tile(
            mA[None, None, bidz],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        gB = cute.local_tile(
            mB[None, None, bidz],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        gC = cute.local_tile(
            mC[None, None, bidz],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, 1, None),
        )
        print("mA layout", mA.layout)
        print("mB layout", mB.layout)
        print("gA layout", gA.layout)
        print("gB layout", gB.layout)

        # By default, if the tensor k mode does not divide into the tile k
        # size, then last tiles in the k dimension are irregular.
        # Instead, make the first tiles irregular when k is irregular.
        # This allows us to handle the irregular tile first to avoid
        # checking for this condition within the mainloop.

        # input is 16B aligned
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
        gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

        # ///////////////////////////////////////////////////////////////////////////////
        # Create shared memory buffers and get the appropriate fragments for this thread.
        # sA:   (BLK_M, BLK_K, PIPE)       , sB:   (BLK_N, BLK_K, PIPE)
        # tAgA: (CPY, CPY_M, CPY_K, k)     , tBgB: (CPY, CPY_N, CPY_K, k)
        # tAsA: (CPY, CPY_M, CPY_K, PIPE)  , tBsB: (CPY, CPY_N, CPY_K, PIPE)
        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory buffer
        smem = cutlass.utils.SmemAllocator()

        sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
        sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)

        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)


        # ///////////////////////////////////////////////////////////////////////////////
        # Prefetch Prologue
        # ///////////////////////////////////////////////////////////////////////////////
        # Clear the smem tiles to account for predicated off loads
        # tAsA.fill(0)
        # tBsB.fill(0)
        cute.arch.sync_threads()
        # Start async loads for the first k-tile. Here we take care of the k residue
        # via if/else check along the k dimension. Because we shifted the identity tensor
        # by the residue_k and because the identity tensor is a counting tensor, the
        # values of any identity tensor element that is poison is less than -1
        num_smem_stages = cute.size(tAsA, mode=[3])
        print("num_smem_stages", num_smem_stages)
        k_tile_count = cute.size(tAgA, mode=[3])

        k_tile_index = cutlass.Int32(0)

        cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index], tAsA[None, None, None, 0])
        cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index], tBsB[None, None, None, 0])

        k_tile_index = k_tile_index + 1
        cute.arch.cp_async_commit_group()

        # ///////////////////////////////////////////////////////////////////////////////
        # Tile MMA compute thread partitions and allocate accumulators
        # ///////////////////////////////////////////////////////////////////////////////
        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        
        print("tCsA layout", tCsA.layout)
        print("tCsB layout", tCsB.layout)

        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgC)
        # Clear the accumulator
        tCrC.fill(0.0)

        # ///////////////////////////////////////////////////////////////////////////////
        # Copy Atom A/B retiling
        # ///////////////////////////////////////////////////////////////////////////////

        # Create the copy atoms for the copy from shared memory to register
        atom_copy_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                self.a_major_mode != cutlass.utils.LayoutEnum.ROW_MAJOR, 4
            ),
            mA.element_type,
        )
        atom_copy_s2r_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                self.b_major_mode != cutlass.utils.LayoutEnum.ROW_MAJOR, 4
            ),
            mB.element_type,
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
        print("tCrA_copy_view layout",tCrA_copy_view.layout)
        print("tCrB_copy_view layout",tCrB_copy_view.layout)


        # Current pipe index in smem to read from / write to

        # tCsA_p = tCsA_copy_view[None, None, None, 0]
        # tCsB_p = tCsB_copy_view[None, None, None, 0]

        # # # ///////////////////////////////////////////////////////////////////////////////
        # # # PREFETCH register pipeline
        # # # ///////////////////////////////////////////////////////////////////////////////
        # num_k_block = cute.size(tCrA, mode=[2])
        # if num_k_block > 1:
        #     # Wait until our first prefetched tile is loaded in
        #     cute.arch.cp_async_wait_group(0)
        #     cute.arch.sync_threads()
        #     # Prefetch the first k-block rmem from the first k-tile
        #     cute.copy(
        #         tiled_copy_s2r_A,
        #         tCsA_p[None, None, 0],
        #         tCrA_copy_view[None, None, 0],
        #     )
        #     cute.copy(
        #         tiled_copy_s2r_B,
        #         tCsB_p[None, None, 0],
        #         tCrB_copy_view[None, None, 0],
        #     )

        # # # ///////////////////////////////////////////////////////////////////////////////
        # # # Mainloop

        # for k_tile in cutlass.range_dynamic(k_tile_count, unroll=1):
        #     if k_tile < k_tile_count - 1:
        #         cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index], tAsA[None, None, None, (k_tile+1) & 1])
        #         cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index], tBsB[None, None, None, (k_tile+1) & 1])
        #         k_tile_index = k_tile_index + 1
        #         cute.arch.cp_async_commit_group()

        #     for k_block in range(num_k_block):
        #         # Load A, B from shared memory to registers for k_block 
        #         if k_block==num_k_block-1:
        #             tCsA_p = tCsA_copy_view[None, None, None, (k_tile+1) & 1]
        #             tCsB_p = tCsB_copy_view[None, None, None, (k_tile+1) & 1]
        #             cute.arch.cp_async_wait_group(0)
        #             cute.arch.sync_threads()
        #         next_k_block = (k_block+1)%num_k_block
        #         cute.copy(
        #             tiled_copy_s2r_A,
        #             tCsA_p[None, None, next_k_block],
        #             tCrA_copy_view[None, None, next_k_block],
        #         )
        #         cute.copy(
        #             tiled_copy_s2r_B,
        #             tCsB_p[None, None, next_k_block],
        #             tCrB_copy_view[None, None, next_k_block],
        #         )

        #         cute.gemm(
        #             tiled_mma,
        #             tCrC,
        #             tCrA[None, None, k_block],
        #             tCrB[None, None, k_block],
        #             tCrC,
        #         )

        for k_tile in cutlass.range_dynamic(k_tile_count, unroll=1):
            if k_tile < k_tile_count - 1:
                cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index], tAsA[None, None, None, (k_tile+1) & 1])
                cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index], tBsB[None, None, None, (k_tile+1) & 1])
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(1)
                cute.arch.sync_threads()
            tCsA_p = tCsA_copy_view[None, None, None, (k_tile) & 1]
            tCsB_p = tCsB_copy_view[None, None, None, (k_tile) & 1]
            for k in cutlass.range_dynamic(cute.size(tCrA, mode=[2])):
                # Load A, B from shared memory to registers for k 
                cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, k], tCrA_copy_view[None, None, k])
                cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, k], tCrB_copy_view[None, None, k])
                cute.gemm(tiled_mma, tCrC, tCrA[None, None, k], tCrB[None, None, k], tCrC)
                    
        
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        cute.autovec_copy(tCrC, tCgC)


    def _make_smem_layout_AB(self, dtype, major_mode, copy_bits, smem_tiler):
        major_mode_size = (
            smem_tiler[1] if major_mode == cutlass.utils.LayoutEnum.ROW_MAJOR else smem_tiler[0]
        )
        major_mode_size = 64 if major_mode_size >= 64 else major_mode_size

        swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)

        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == cutlass.utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            layout_atom_outer,
        )
        layout = cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))
        return layout

    def _make_smem_layout_C(self, dtype, major_mode, copy_bits, smem_tiler):
        major_mode_size = (
            smem_tiler[1] if major_mode == cutlass.utils.LayoutEnum.ROW_MAJOR else smem_tiler[0]
        )

        swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)

        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == cutlass.utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 4),
            0,
            layout_atom_outer,
        )

        # Due to the thread layout of the mma, remove swizzle in C to
        # prevent shared memory fragments owned by an single thread from
        # holding swizzles
        if major_mode == cutlass.utils.LayoutEnum.COL_MAJOR:
            layout_atom = cute.make_composed_layout(
                cute.make_swizzle(0, 3, 4), 0, layout_atom_outer
            )
        layout = cute.tile_to_shape(
            layout_atom,
            smem_tiler,
            (0, 1),
        )
        return layout

    def _make_gmem_tiled_copy_AB(self, atom_copy, dtype, major_mode, copy_bits):
        copy_elems = copy_bits // dtype.width
        shape_dim_1 = cute.size(self.bK) // copy_elems
        # thread layout for copy
        thread_layout = cute.make_layout(
            (self.num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
        )
        if major_mode != cutlass.utils.LayoutEnum.ROW_MAJOR:
            shape_dim_0 = cute.size(self.bM) // copy_elems
            thread_layout = cute.make_layout(
                (shape_dim_0, self.num_threads // shape_dim_0), stride=(1, shape_dim_0)
            )
        # Value layout for copy
        value_layout = (
            cute.make_layout((1, copy_elems))
            if major_mode == cutlass.utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    def _make_gmem_tiled_copy_C(self, atom_copy, dtype, major_mode, copy_bits):
        copy_elems = copy_bits // dtype.width
        shape_dim_1 = cute.size(self.bN) // copy_elems
        # thread layout for copy
        thread_layout = cute.make_layout(
            (self.num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
        )
        if major_mode != cutlass.utils.LayoutEnum.ROW_MAJOR:
            shape_dim_0 = cute.size(self.bM) // copy_elems
            thread_layout = cute.make_layout(
                (shape_dim_0, self.num_threads // shape_dim_0), stride=(1, shape_dim_0)
            )
        value_layout = (
            cute.make_layout((1, copy_elems))
            if major_mode == cutlass.utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        tiler_mn, layout_tv = cute.make_layout_tv(thread_layout, value_layout)
        return cute.make_tiled_copy(atom_copy, layout_tv, tiler_mn)


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
a = create_and_permute_tensor(
    L, M, K, a_major == "m", cutlass_torch.dtype(ab_dtype)
)
b = create_and_permute_tensor(
    L, N, K, b_major == "n", cutlass_torch.dtype(ab_dtype)
)
c = create_and_permute_tensor(L, M, N, c_major == "m", cutlass_torch.dtype(c_dtype))

print("a.shape", a.shape)
print("b.shape", b.shape)
print("c.shape", c.shape)

print("a.strides", a.stride())
print("b.strides", b.stride())
print("c.strides", c.stride())

c_ref = torch.einsum("mkl,nkl->mnl", a, b).to(cutlass_torch.dtype(c_dtype))

tensor_op_gemm = TensorOpGemm(
    ab_dtype,
    c_dtype,
    acc_dtype,
    atom_layout_mnk,
)

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
# gemm_f16f16f16_nn_ = cute.compile(tensor_op_gemm, mA, mB, mC)
# tensor_op_gemm_ = cute.compile(tensor_op_gemm, mA, mB, mC)

compilation_time = time.time() - start_time
print(f"Compilation time: {compilation_time:.4f} seconds")

gemm_f16f16f16_nn_(mA, mB, mC)
# Verify correctness - compare with torch reference
# Fixed: No need to transpose B since both A and B are k-major
# c_ref = torch.matmul(a.to(torch.float16), b.to(torch.float16).T).to(torch.float16)

# diff = calc_diff(c_torch.cpu(), c_ref)
print("c_torch", c.cpu()[:10,:10,0])
print("c_ref", c_ref[:10,:10,0])

print("c_torch.shape", c.shape)
print("c_ref.shape", c_ref.shape)

print("c_torch.stride", c.stride())
print("c_ref.stride", c_ref.stride())

# print(f"diff: {diff}")
# assert diff < 1e-5
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
    
    




