import pytest

import numpy as np

import pystencils.config
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

    config = pystencils.config.CreateKernelConfig(cpu_vectorize_info={'instruction_set': instruction_set})
    ast = ps.create_kernel(update_rule, config=config)
    kernel = ast.compile()
    kernel(f=arr)
    np.testing.assert_equal(arr, 2)


@pytest.mark.parametrize('dtype', ('float32', 'float64'))
@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
def test_vectorized_abs(instruction_set, dtype):
    """Some instructions sets have abs, some don't.
       Furthermore, the special treatment of unary minus makes this data type-sensitive too.
    """
    arr = np.ones((2 ** 2 + 2, 2 ** 3 + 2), dtype=dtype)
    arr[-3:, :] = -1

    f, g = ps.fields(f=arr, g=arr)
    update_rule = [ps.Assignment(g.center(), sp.Abs(f.center()))]

    config = pystencils.config.CreateKernelConfig(cpu_vectorize_info={'instruction_set': instruction_set})
    ast = ps.create_kernel(update_rule, config=config)

    func = ast.compile()
    dst = np.zeros_like(arr)
    func(g=dst, f=arr)
    np.testing.assert_equal(np.sum(dst[1:-1, 1:-1]), 2 ** 2 * 2 ** 3)


@pytest.mark.parametrize('dtype', ('float32', 'float64'))
@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
def test_strided(instruction_set, dtype):
    f, g = ps.fields(f"f, g : {dtype}[2D]")
    update_rule = [ps.Assignment(g[0, 0], f[0, 0] + f[-1, 0] + f[1, 0] + f[0, 1] + f[0, -1] + 42.0)]
    if 'storeS' not in get_vector_instruction_set(dtype, instruction_set) \
            and instruction_set not in ['avx512', 'rvv'] and not instruction_set.startswith('sve'):
        with pytest.warns(UserWarning) as warn:
            config = pystencils.config.CreateKernelConfig(cpu_vectorize_info={'instruction_set': instruction_set},
                                                          default_number_float=dtype)
            ast = ps.create_kernel(update_rule, config=config)
            assert 'Could not vectorize loop' in warn[0].message.args[0]
    else:
        with pytest.warns(None) as warn:
            config = pystencils.config.CreateKernelConfig(cpu_vectorize_info={'instruction_set': instruction_set},
                                                          default_number_float=dtype)
            ast = ps.create_kernel(update_rule, config=config)
            assert len(warn) == 0

    # ps.show_code(ast)
    func = ast.compile()
    ref_config = pystencils.config.CreateKernelConfig(default_number_float=dtype)
    ref_func = ps.create_kernel(update_rule, config=ref_config).compile()

    # For some reason other array creations fail on the emulated ppc pipeline
    size = (25, 19)
    arr = np.zeros(size).astype(dtype)
    for i in range(size[0]):
        for j in range(size[1]):
            arr[i, j] = i * j

    dst = np.zeros_like(arr, dtype=dtype)
    ref = np.zeros_like(arr, dtype=dtype)

    func(g=dst, f=arr)
    ref_func(g=ref, f=arr)

    # print("dst: ", dst)
    # print("np array: ", arr)

    np.testing.assert_almost_equal(dst[1:-1, 1:-1], ref[1:-1, 1:-1], 13 if dtype == 'float64' else 5)


@pytest.mark.parametrize('dtype', ('float32', 'float64'))
@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
@pytest.mark.parametrize('gl_field, gl_kernel', [(1, 0), (0, 1), (1, 1)])
def test_alignment_and_correct_ghost_layers(gl_field, gl_kernel, instruction_set, dtype):
    domain_size = (128, 128)
    dh = ps.create_data_handling(domain_size, periodicity=(True, True), default_target=Target.CPU)
    src = dh.add_array("src", values_per_cell=1, dtype=dtype, ghost_layers=gl_field, alignment=True)
    dh.fill(src.name, 1.0, ghost_layers=True)
    dst = dh.add_array("dst", values_per_cell=1, dtype=dtype, ghost_layers=gl_field, alignment=True)
    dh.fill(dst.name, 1.0, ghost_layers=True)

    update_rule = ps.Assignment(dst[0, 0], src[0, 0])
    opt = {'instruction_set': instruction_set, 'assume_aligned': True,
           'nontemporal': True, 'assume_inner_stride_one': True}
    config = pystencils.config.CreateKernelConfig(target=dh.default_target,
                                                  cpu_vectorize_info=opt, ghost_layers=gl_kernel)
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


# TODO move to vectorise
@pytest.mark.parametrize('instruction_set',
                         sorted(set(supported_instruction_sets) - {test_vectorization.instruction_set}))
@pytest.mark.parametrize('function',
                         [f for f in test_vectorization.__dict__ if f.startswith('test_') and f not in ['test_hardware_query', 'test_aligned_and_nt_stores']])
def test_vectorization_other(instruction_set, function):
    test_vectorization.__dict__[function](instruction_set)


@pytest.mark.parametrize('dtype', ('float32', 'float64'))
@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
@pytest.mark.parametrize('field_layout', ('fzyx', 'zyxf'))
def test_square_root(dtype, instruction_set, field_layout):
    config = pystencils.config.CreateKernelConfig(data_type=dtype,
                                                  default_number_float=dtype,
                                                  cpu_vectorize_info={'instruction_set': instruction_set,
                                                                      'assume_inner_stride_one': True,
                                                                      'assume_aligned': False,
                                                                      'nontemporal': False})

    src_field = ps.Field.create_generic('pdfs', 2, dtype, index_dimensions=1, layout=field_layout, index_shape=(9,))

    eq = [ps.Assignment(sp.Symbol("xi"), sum(src_field.center_vector)),
          ps.Assignment(sp.Symbol("xi_2"), sp.Symbol("xi") * sp.sqrt(src_field.center))]

    ast = ps.create_kernel(eq, config=config)
    ast.compile()
    code = ps.get_code_str(ast)
    print(code)


@pytest.mark.parametrize('dtype', ('float32', 'float64'))
@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
@pytest.mark.parametrize('padding', (True, False))
def test_square_root_2(dtype, instruction_set, padding):
    x, y = sp.symbols("x y")
    src = ps.fields(f"src: {dtype}[2D]", layout='fzyx')

    up = ps.Assignment(src[0, 0], 1 / x + sp.sqrt(y * 0.52 + x ** 2))

    cpu_vec = {'instruction_set': instruction_set, 'assume_inner_stride_one': True,
               'assume_sufficient_line_padding': padding,
               'assume_aligned': True}

    config = ps.CreateKernelConfig(data_type=dtype, default_number_float=dtype, cpu_vectorize_info=cpu_vec)
    ast = ps.create_kernel(up, config=config)
    ast.compile()

    code = ps.get_code_str(ast)
    print(code)


@pytest.mark.parametrize('dtype', ('float32', 'float64'))
@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
@pytest.mark.parametrize('padding', (True, False))
def test_pow(dtype, instruction_set, padding):
    config = pystencils.config.CreateKernelConfig(data_type=dtype,
                                                  default_number_float=dtype,
                                                  cpu_vectorize_info={'instruction_set': instruction_set,
                                                                      'assume_inner_stride_one': True,
                                                                      'assume_sufficient_line_padding': padding,
                                                                      'assume_aligned': False, 'nontemporal': False})

    src_field = ps.Field.create_generic('pdfs', 2, dtype, index_dimensions=1, layout='fzyx', index_shape=(9,))

    eq = [ps.Assignment(sp.Symbol("xi"), sum(src_field.center_vector)),
          ps.Assignment(sp.Symbol("xi_2"), sp.Symbol("xi") * sp.Pow(src_field.center, 0.5))]

    ast = ps.create_kernel(eq, config=config)
    ast.compile()
    code = ps.get_code_str(ast)

    print(code)


@pytest.mark.parametrize('dtype', ('float32', 'float64'))
@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
@pytest.mark.parametrize('padding', (True, False))
def test_issue62(dtype, instruction_set, padding):
    opt = {'instruction_set': instruction_set, 'assume_aligned': True,
           'assume_inner_stride_one': True,
           'assume_sufficient_line_padding': padding}

    dx = sp.Symbol("dx")
    dy = sp.Symbol("dy")
    src, dst, rhs = ps.fields(f"src, src_tmp, rhs: {dtype}[2D]", layout='fzyx')

    up = ps.Assignment(src[0, 0], ((dy ** 2 * (src[1, 0] + src[-1, 0]))
                                   + (dx ** 2 * (src[0, 1] + src[0, -1]))
                                   - (rhs[0, 0] * dx ** 2 * dy ** 2)) / (2 * (dx ** 2 + dy ** 2)))

    config = ps.CreateKernelConfig(data_type=dtype,
                                   default_number_float=dtype,
                                   cpu_vectorize_info=opt)

    ast = ps.create_kernel(up, config=config)
    ast.compile()
    code = ps.get_code_str(ast)

    print(code)

    assert 'pow' not in code


@pytest.mark.parametrize('dtype', ('float32', 'float64'))
@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
def test_div_and_unevaluated_expr(dtype, instruction_set):
    opt = {'instruction_set': instruction_set, 'assume_aligned': True, 'assume_inner_stride_one': True,
           'assume_sufficient_line_padding': False}

    x, y, z = sp.symbols("x y z")
    rhs = (-4 * x ** 4 * y ** 2 * z ** 2 + (x ** 4 * y ** 2 / 3) + (x ** 4 * z ** 2 / 3)) / x ** 3

    src = ps.fields(f"src: {dtype}[2D]", layout='fzyx')

    up = ps.Assignment(src[0, 0], rhs)

    config = ps.CreateKernelConfig(data_type=dtype,
                                   default_number_float=dtype,
                                   cpu_vectorize_info=opt)

    ast = ps.create_kernel(up, config=config)
    code = ps.get_code_str(ast)
    # print(code)

    ast.compile()

    assert 'pow' not in code


# TODO this test case needs a complete rework of the vectoriser. The reason is that the vectoriser does not
# TODO vectorise symbols at the moment because they could be strides or field sizes, thus involved in pointer arithmetic
# TODO This means that the vectoriser only works if fields are involved on the rhs.
# @pytest.mark.parametrize('dtype', ('float32', 'float64'))
# @pytest.mark.parametrize('instruction_set', supported_instruction_sets)
# def test_vectorised_symbols(dtype, instruction_set):
#     opt = {'instruction_set': instruction_set, 'assume_aligned': True, 'assume_inner_stride_one': True,
#            'assume_sufficient_line_padding': False}
#
#     x, y, z = sp.symbols("x y z")
#     rhs = 1 / x ** 2 * (x + y)
#
#     src = ps.fields(f"src: {dtype}[2D]", layout='fzyx')
#
#     up = ps.Assignment(src[0, 0], rhs)
#
#     config = ps.CreateKernelConfig(data_type=dtype,
#                                    default_number_float=dtype,
#                                    cpu_vectorize_info=opt)
#
#     ast = ps.create_kernel(up, config=config)
#     code = ps.get_code_str(ast)
#     print(code)
#
#     ast.compile()
#
#     assert 'pow' not in code
