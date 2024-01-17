import pytest

from pystencils import Target

from pystencils.nbackend.ast import *
from pystencils.nbackend.constraints import PsKernelConstraint
from pystencils.nbackend.typed_expressions import *
from pystencils.nbackend.arrays import PsLinearizedArray, PsArrayBasePointer, PsArrayAccess
from pystencils.nbackend.types.quick import *

import numpy as np

from pystencils.cpu.cpujit import compile_and_load

def test_pairwise_addition():
    idx_type = SInt(64)

    u = PsLinearizedArray("u", Fp(64, const=True), (..., ...), (..., ...), index_dtype=idx_type)
    v = PsLinearizedArray("v", Fp(64), (..., ...), (..., ...), index_dtype=idx_type)

    u_data = PsArrayBasePointer("u_data", u)
    v_data = PsArrayBasePointer("v_data", v)

    loop_ctr = PsTypedVariable("ctr", idx_type)
    
    zero = PsTypedConstant(0, idx_type)
    one = PsTypedConstant(1, idx_type)
    two = PsTypedConstant(2, idx_type)

    update = PsAssignment(
        PsLvalueExpr(PsArrayAccess(v_data, loop_ctr)),
        PsExpression(PsArrayAccess(u_data, two * loop_ctr) + PsArrayAccess(u_data, two * loop_ctr + one))
    )

    loop = PsLoop(
        PsSymbolExpr(loop_ctr),
        PsExpression(zero),
        PsExpression(v.shape[0]),
        PsExpression(one),
        PsBlock([update])
    )

    func = PsKernelFunction(PsBlock([loop]), target=Target.CPU)

    sizes_constraint = PsKernelConstraint(
        u.shape[0].eq(2 * v.shape[0]),
        "Array `u` must have twice the length of array `v`"
    )

    func.add_constraints(sizes_constraint)

    kernel = compile_and_load(func)

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
    
