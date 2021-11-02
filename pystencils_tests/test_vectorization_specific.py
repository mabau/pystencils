import pytest

import numpy as np
import sympy as sp

import pystencils as ps
from pystencils.backends.simd_instruction_sets import (get_cacheline_size, get_supported_instruction_sets,
                                                       get_vector_instruction_set)
from . import test_vectorization
from pystencils.enums import Target

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

    config = ps.CreateKernelConfig(cpu_vectorize_info={'instruction_set': instruction_set})
    ast = ps.create_kernel(update_rule, config=config)
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

    config = ps.CreateKernelConfig(cpu_vectorize_info={'instruction_set': instruction_set})
    ast = ps.create_kernel(update_rule, config=config)

    func = ast.compile()
    dst = np.zeros_like(arr)
    func(g=dst, f=arr)
    np.testing.assert_equal(np.sum(dst[1:-1, 1:-1]), 2 ** 2 * 2 ** 3)


@pytest.mark.parametrize('dtype', ('float', 'double'))
@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
def test_strided(instruction_set, dtype):
    f, g = ps.fields(f"f, g : float{64 if dtype == 'double' else 32}[2D]")
    update_rule = [ps.Assignment(g[0, 0], f[0, 0] + f[-1, 0] + f[1, 0] + f[0, 1] + f[0, -1] + 42.0)]
    if 'storeS' not in get_vector_instruction_set(dtype, instruction_set) and not instruction_set in ['avx512', 'rvv'] and not instruction_set.startswith('sve'):
        with pytest.warns(UserWarning) as warn:
            config = ps.CreateKernelConfig(cpu_vectorize_info={'instruction_set': instruction_set})
            ast = ps.create_kernel(update_rule, config=config)
            assert 'Could not vectorize loop' in warn[0].message.args[0]
    else:
        with pytest.warns(None) as warn:
            config = ps.CreateKernelConfig(cpu_vectorize_info={'instruction_set': instruction_set})
            ast = ps.create_kernel(update_rule, config=config)
            assert len(warn) == 0
    func = ast.compile()
    ref_func = ps.create_kernel(update_rule).compile()

    arr = np.random.random((23 + 2, 17 + 2)).astype(np.float64 if dtype == 'double' else np.float32)
    dst = np.zeros_like(arr, dtype=np.float64 if dtype == 'double' else np.float32)
    ref = np.zeros_like(arr, dtype=np.float64 if dtype == 'double' else np.float32)

    func(g=dst, f=arr)
    ref_func(g=ref, f=arr)
    np.testing.assert_almost_equal(dst, ref, 13 if dtype == 'double' else 5)


@pytest.mark.parametrize('dtype', ('float', 'double'))
@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
@pytest.mark.parametrize('gl_field, gl_kernel', [(1, 0), (0, 1), (1, 1)])
def test_alignment_and_correct_ghost_layers(gl_field, gl_kernel, instruction_set, dtype):
    dtype = np.float64 if dtype == 'double' else np.float32

    domain_size = (128, 128)
    dh = ps.create_data_handling(domain_size, periodicity=(True, True), default_target=Target.CPU)
    src = dh.add_array("src", values_per_cell=1, dtype=dtype, ghost_layers=gl_field, alignment=True)
    dh.fill(src.name, 1.0, ghost_layers=True)
    dst = dh.add_array("dst", values_per_cell=1, dtype=dtype, ghost_layers=gl_field, alignment=True)
    dh.fill(dst.name, 1.0, ghost_layers=True)

    update_rule = ps.Assignment(dst[0, 0], src[0, 0])
    opt = {'instruction_set': instruction_set, 'assume_aligned': True,
           'nontemporal': True, 'assume_inner_stride_one': True}
    config = ps.CreateKernelConfig(target=dh.default_target, cpu_vectorize_info=opt, ghost_layers=gl_kernel)
    ast = ps.create_kernel(update_rule, config=config)
    kernel = ast.compile()
    if gl_kernel != gl_field:
        with pytest.raises(ValueError):
            dh.run_kernel(kernel)
    else:
        dh.run_kernel(kernel)


@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
def test_cacheline_size(instruction_set):
    cacheline_size = get_cacheline_size(instruction_set)
    if cacheline_size is None and instruction_set in ['sse', 'avx', 'avx512', 'rvv']:
        pytest.skip()
    instruction_set = get_vector_instruction_set('double', instruction_set)
    vector_size = instruction_set['bytes']
    assert 8 < cacheline_size < 0x100000, "Cache line size is implausible"
    if type(vector_size) is int:
        assert cacheline_size % vector_size == 0, "Cache line size should be multiple of vector size"
    assert cacheline_size & (cacheline_size - 1) == 0, "Cache line size is not a power of 2"


# test_vectorization is not parametrized because it is supposed to run without pytest, so we parametrize it here
@pytest.mark.parametrize('instruction_set',
                         sorted(set(supported_instruction_sets) - {test_vectorization.instruction_set}))
@pytest.mark.parametrize('function',
                         [f for f in test_vectorization.__dict__ if f.startswith('test_') and f != 'test_hardware_query'])
def test_vectorization_other(instruction_set, function):
    test_vectorization.__dict__[function](instruction_set)


@pytest.mark.parametrize('dtype', ('float', 'double'))
@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
@pytest.mark.parametrize('field_layout', ('fzyx', 'zyxf'))
def test_square_root(dtype, instruction_set, field_layout):
    config = ps.CreateKernelConfig(data_type=dtype,
                                   cpu_vectorize_info={'instruction_set': instruction_set,
                                                       'assume_inner_stride_one': True,
                                                       'assume_aligned': False, 'nontemporal': False})

    src_field = ps.Field.create_generic('pdfs', 2, dtype, index_dimensions=1, layout=field_layout, index_shape=(9,))

    eq = [ps.Assignment(sp.Symbol("xi"), sum(src_field.center_vector)),
          ps.Assignment(sp.Symbol("xi_2"), sp.Symbol("xi") * sp.sqrt(src_field.center))]

    ps.create_kernel(eq, config=config).compile()
