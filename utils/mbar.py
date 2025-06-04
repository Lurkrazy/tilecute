
from cutlass.cutlass_dsl import CuTeDSL, T, if_generate, dsl_user_op

from cutlass._mlir.dialects import nvvm
from cutlass._mlir import ir

# from ..typing import Pointer, Int, Boolean, Int32
# from ...impl_utils import check_value_in

from cutlass.cute.typing import Pointer, Int, Boolean, Int32
from cutlass.impl_utils import check_value_in


@dsl_user_op
def mbarrier_expect_tx(
    mbar_ptr: Pointer, bytes: Int, *, loc=None, ip=None
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
    mbar_llvm_ptr = mbar_ptr.llvm_ptr
    space = nvvm.MBarrierSpaceKind.CTA

    nvvm.mbarrier_txn(
        mbar_llvm_ptr,
        Int32(bytes).ir_value(loc=loc, ip=ip),
        kind=nvvm.MBarrierTxnKind.EXPECT_TX,
        space=space,
        loc=loc,
        ip=ip,
    )
