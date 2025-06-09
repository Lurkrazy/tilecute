
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

@dsl_user_op
def mbarrier_wait(mbar_ptr: Pointer, phase: Int, timeout_ns: Int = 10000000, *, loc=None, ip=None) -> None:
    """
    Waits on a mbarrier with a specified phase.

    :param mbar_ptr: A pointer to the mbarrier in SMEM
    :type mbar_ptr:  Pointer
    :param phase:    The phase to wait for (either 0 or 1)
    :type phase:     Int
    """
    arch = CuTeDSL._get_dsl().envar.arch
    check_value_in(arch, ["sm_90", "sm_90a", "sm_100a"], "arch")

    # timeout_ns = 10000000
    # This NVVM Op is a spin-loop wrapping the mbarrier.try_wait.parity.shared.b64 PTX
    # The timeout in ns only applies to the latter and this call is truly blocking
    nvvm.mbarrier_try_wait_parity_shared(
        mbar_ptr.llvm_ptr,
        Int32(phase).ir_value(loc=loc, ip=ip),
        Int32(timeout_ns).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )