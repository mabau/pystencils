import pytest

from pystencils import Target

# from pystencils.backend.constraints import PsKernelParamsConstraint
from pystencils.backend.memory import PsSymbol, PsBuffer
from pystencils.backend.constants import PsConstant

from pystencils.backend.ast.expressions import PsBufferAcc, PsExpression
from pystencils.backend.ast.structural import PsAssignment, PsBlock, PsLoop
from pystencils.backend.kernelfunction import KernelFunction

from pystencils.types.quick import SInt, Fp
from pystencils.backend.jit import LegacyCpuJit

import numpy as np


@pytest.mark.xfail(reason="Fails until constraints are reimplemented")
def test_pairwise_addition():
    idx_type = SInt(64)

    u = PsBuffer("u", Fp(64, const=True), (...,), (...,), index_dtype=idx_type)
    v = PsBuffer("v", Fp(64), (...,), (...,), index_dtype=idx_type)

    u_data = PsArrayBasePointer("u_data", u)
    v_data = PsArrayBasePointer("v_data", v)

    loop_ctr = PsExpression.make(PsSymbol("ctr", idx_type))
    
    zero = PsExpression.make(PsConstant(0, idx_type))
    one = PsExpression.make(PsConstant(1, idx_type))
    two = PsExpression.make(PsConstant(2, idx_type))

    update = PsAssignment(
        PsBufferAcc(v_data, loop_ctr),
        PsBufferAcc(u_data, two * loop_ctr) + PsBufferAcc(u_data, two * loop_ctr + one)
    )

    loop = PsLoop(
        loop_ctr,
        zero,
        PsExpression.make(v.shape[0]),
        one,
        PsBlock([update])
    )

    func = KernelFunction(PsBlock([loop]), Target.CPU, "kernel", set())

    # sizes_constraint = PsKernelParamsConstraint(
    #     u.shape[0].eq(2 * v.shape[0]),
    #     "Array `u` must have twice the length of array `v`"
    # )

    # func.add_constraints(sizes_constraint)

    jit = LegacyCpuJit()
    kernel = jit.compile(func)

    #   Positive case
    N = 21
    u_arr = np.arange(2 * N, dtype=np.float64)
    v_arr = np.zeros((N,), dtype=np.float64)

    assert u_arr.shape[0] == 2 * v_arr.shape[0]

    kernel(u=u_arr, v=v_arr)

    v_expected = np.zeros_like(v_arr)
    for i in range(N):
        v_expected[i] = u_arr[2 * i] + u_arr[2*i + 1]

    np.testing.assert_allclose(v_arr, v_expected)

    #   Negative case - mismatched array sizes
    u_arr = np.zeros((N + 2,), dtype=np.float64)
    v_arr = np.zeros((N,), dtype=np.float64)

    with pytest.raises(ValueError):
        kernel(u=u_arr, v=v_arr)

    #   Negative case - mismatched types
    u_arr = np.arange(2 * N, dtype=np.float64)
    v_arr = np.zeros((N,), dtype=np.float32)

    with pytest.raises(TypeError):
        kernel(u=u_arr, v=v_arr)
    
