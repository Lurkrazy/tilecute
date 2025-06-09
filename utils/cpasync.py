
from cutlass.cutlass_dsl import CuTeDSL, T, if_generate, dsl_user_op

from cutlass._mlir.dialects import nvvm, cute_nvgpu
from cutlass._mlir import ir

# from ..typing import Pointer, Int, Boolean, Int32
# from ...impl_utils import check_value_in
from cutlass.cute import core
from cutlass.cute.typing import Pointer, Int, Boolean, Int32, Int16
from cutlass.impl_utils import check_value_in


@dsl_user_op
def tma_load(
    tma_desc, mbar: Pointer, smem_ptr: Pointer, crd: tuple[Int, ...], *, loc=None, ip=None
) -> None:
    """
    Arrives on an mbarrier.

    :param mbar_ptr:                 A pointer to the mbarrier in SMEM
    :type mbar_ptr:                  Pointer
    :param bytes:                    The number of transaction bytes
    :type bytes:                     Int
    """
    arch = CuTeDSL._get_dsl().envar.arch
    check_value_in(arch, ["sm_90", "sm_90a", "sm_100a"], "arch")

    # build with NV_CONTRIB would set use_intrinsic to True, which would lead to compile error
    # cute_nvgpu.arch_copy_SM100_tma_load(
    #     mode = cute_nvgpu.TmaLoadMode.tiled,
    #     num_cta = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 1),
    #     src_desc = tma_desc.value,
    #     dsmem_data_addr = smem_ptr.value,
    #     dsmem_bar_addr = mbar.value,
    #     coord = [Int32(i).ir_value(loc=loc, ip=ip) for i in crd],
    #     offsets = [], 
    #     multicast_mask = None,
    #     loc=loc,
    #     ip=ip
    # )
    
    nvvm.cp_async_bulk_tensor_shared_cluster_global(
        dst_mem = smem_ptr.llvm_ptr,
        tma_descriptor = tma_desc.llvm_ptr,
        coordinates = [Int32(i).ir_value(loc=loc, ip=ip) for i in crd],
        mbar = mbar.llvm_ptr,
        im2col_offsets = [],
        load_mode = nvvm.CpAsyncBulkTensorLoadMode.TILE,
        group = nvvm.Tcgen05GroupKind.CTA_1,
        use_intrinsic = False, # set to True would lead to compile error
        loc=loc,
        ip=ip
    )
