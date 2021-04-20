import pytest

import numpy as np
import sympy as sp

import pystencils as ps
from pystencils.backends.simd_instruction_sets import (get_cacheline_size, get_supported_instruction_sets,
                                                       get_vector_instruction_set)
from pystencils.data_types import cast_func, VectorType

supported_instruction_sets = get_supported_instruction_sets() if get_supported_instruction_sets() else []


@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
def test_vectorisation_varying_arch(instruction_set):
    shape = (9, 9, 3)
    arr = np.ones(shape, order='f')

    @ps.kernel
    def update_rule(s):
        f = ps.fields("f(3) : [2D]", f=arr)
        s.tmp0 @= f(0)
        s.tmp1 @= f(1)
        s.tmp2 @= f(2)
        f0, f1, f2 = f(0), f(1), f(2)
        f0 @= 2 * s.tmp0
        f1 @= 2 * s.tmp0
        f2 @= 2 * s.tmp0

    ast = ps.create_kernel(update_rule, cpu_vectorize_info={'instruction_set': instruction_set})
    kernel = ast.compile()
    kernel(f=arr)
    np.testing.assert_equal(arr, 2)


@pytest.mark.parametrize('dtype', ('float', 'double'))
@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
def test_vectorized_abs(instruction_set, dtype):
    """Some instructions sets have abs, some don't.
       Furthermore, the special treatment of unary minus makes this data type-sensitive too.
    """
    arr = np.ones((2 ** 2 + 2, 2 ** 3 + 2), dtype=np.float64 if dtype == 'double' else np.float32)
    arr[-3:, :] = -1

    f, g = ps.fields(f=arr, g=arr)
    update_rule = [ps.Assignment(g.center(), sp.Abs(f.center()))]

    ast = ps.create_kernel(update_rule, cpu_vectorize_info={'instruction_set': instruction_set})

    func = ast.compile()
    dst = np.zeros_like(arr)
    func(g=dst, f=arr)
    np.testing.assert_equal(np.sum(dst[1:-1, 1:-1]), 2 ** 2 * 2 ** 3)


@pytest.mark.parametrize('dtype', ('float', 'double'))
@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
@pytest.mark.parametrize('gl_field, gl_kernel', [(1, 0), (0, 1), (1, 1)])
def test_alignment_and_correct_ghost_layers(gl_field, gl_kernel, instruction_set, dtype):
    dtype = np.float64 if dtype == 'double' else np.float32

    domain_size = (128, 128)
    dh = ps.create_data_handling(domain_size, periodicity=(True, True), default_target='cpu')
    src = dh.add_array("src", values_per_cell=1, dtype=dtype, ghost_layers=gl_field, alignment=True)
    dh.fill(src.name, 1.0, ghost_layers=True)
    dst = dh.add_array("dst", values_per_cell=1, dtype=dtype, ghost_layers=gl_field, alignment=True)
    dh.fill(dst.name, 1.0, ghost_layers=True)

    update_rule = ps.Assignment(dst[0, 0], src[0, 0])
    opt = {'instruction_set': instruction_set, 'assume_aligned': True,
           'nontemporal': True, 'assume_inner_stride_one': True}
    ast = ps.create_kernel(update_rule, target=dh.default_target, cpu_vectorize_info=opt, ghost_layers=gl_kernel)
    kernel = ast.compile()
    if gl_kernel != gl_field:
        with pytest.raises(ValueError):
            dh.run_kernel(kernel)
    else:
        dh.run_kernel(kernel)


@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
def test_cacheline_size(instruction_set):
    cacheline_size = get_cacheline_size(instruction_set)
    if cacheline_size is None:
        pytest.skip()
    instruction_set = get_vector_instruction_set('double', instruction_set)
    vector_size = instruction_set['bytes']
    assert cacheline_size > 8 and cacheline_size < 0x100000, "Cache line size is implausible"
    assert cacheline_size % vector_size == 0, "Cache line size should be multiple of vector size"
    assert cacheline_size & (cacheline_size - 1) == 0, "Cache line size is not a power of 2"


# test_vectorization is not parametrized because it is supposed to run without pytest, so we parametrize it here
from pystencils_tests import test_vectorization
@pytest.mark.parametrize('instruction_set', set(supported_instruction_sets) - set([test_vectorization.instruction_set]))
@pytest.mark.parametrize('function', [f for f in test_vectorization.__dict__ if f.startswith('test_') and f != 'test_hardware_query'])
def test_vectorization_other(instruction_set, function):
    test_vectorization.__dict__[function](instruction_set)
