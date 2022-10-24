import numpy as np

import pytest

import pystencils.config
import sympy as sp

import pystencils as ps
from pystencils.backends.simd_instruction_sets import get_supported_instruction_sets, get_vector_instruction_set
from pystencils.cpu.vectorization import vectorize
from pystencils.enums import Target
from pystencils.transformations import replace_inner_stride_with_one

supported_instruction_sets = get_supported_instruction_sets()
if supported_instruction_sets:
    instruction_set = supported_instruction_sets[-1]
else:
    instruction_set = None


# TODO: Skip tests if no instruction set is available and check all codes if they are really vectorised !
def test_vector_type_propagation(instruction_set=instruction_set):
    a, b, c, d, e = sp.symbols("a b c d e")
    arr = np.ones((2 ** 2 + 2, 2 ** 3 + 2))
    arr *= 10.0

    f, g = ps.fields(f=arr, g=arr)
    update_rule = [ps.Assignment(a, f[1, 0]),
                   ps.Assignment(b, a),
                   ps.Assignment(g[0, 0], b + 3 + f[0, 1])]

    ast = ps.create_kernel(update_rule)
    vectorize(ast, instruction_set=instruction_set)

    # ps.show_code(ast)

    func = ast.compile()
    dst = np.zeros_like(arr)
    func(g=dst, f=arr)
    np.testing.assert_equal(dst[1:-1, 1:-1], 2 * 10.0 + 3)


@pytest.mark.parametrize('openmp', [True, False])
def test_aligned_and_nt_stores(openmp, instruction_set=instruction_set):
    domain_size = (24, 24)
    # create a datahandling object
    dh = ps.create_data_handling(domain_size, periodicity=(True, True), parallel=False, default_target=Target.CPU)

    # fields
    alignment = 'cacheline' if openmp else True
    g = dh.add_array("g", values_per_cell=1, alignment=alignment)
    dh.fill("g", 1.0, ghost_layers=True)
    f = dh.add_array("f", values_per_cell=1, alignment=alignment)
    dh.fill("f", 0.0, ghost_layers=True)
    opt = {'instruction_set': instruction_set, 'assume_aligned': True, 'nontemporal': True,
           'assume_inner_stride_one': True}
    update_rule = [ps.Assignment(f.center(), 0.25 * (g[-1, 0] + g[1, 0] + g[0, -1] + g[0, 1]))]
    config = pystencils.config.CreateKernelConfig(target=dh.default_target, cpu_vectorize_info=opt, cpu_openmp=openmp)
    ast = ps.create_kernel(update_rule, config=config)
    if instruction_set in ['sse'] or instruction_set.startswith('avx'):
        assert 'stream' in ast.instruction_set
        assert 'streamFence' in ast.instruction_set
    if instruction_set in ['neon', 'vsx'] or instruction_set.startswith('sve'):
        assert 'cachelineZero' in ast.instruction_set
    if instruction_set in ['vsx']:
        assert 'storeAAndFlushCacheline' in ast.instruction_set
    for instruction in ['stream', 'streamFence', 'cachelineZero', 'storeAAndFlushCacheline', 'flushCacheline']:
        if instruction in ast.instruction_set:
            assert ast.instruction_set[instruction].split('{')[0] in ps.get_code_str(ast)
    kernel = ast.compile()

    # ps.show_code(ast)

    dh.run_kernel(kernel)
    np.testing.assert_equal(np.sum(dh.cpu_arrays['f']), np.prod(domain_size))


def test_nt_stores_symbolic_size(instruction_set=instruction_set):
    f, g = ps.fields('f, g: [2D]', layout='fzyx')
    update_rule = [ps.Assignment(f.center(), 0.0), ps.Assignment(g.center(), 0.0)]
    opt = {'instruction_set': instruction_set, 'assume_aligned': True, 'nontemporal': True,
           'assume_inner_stride_one': True}
    config = pystencils.config.CreateKernelConfig(target=Target.CPU, cpu_vectorize_info=opt)
    ast = ps.create_kernel(update_rule, config=config)
    # ps.show_code(ast)
    ast.compile()


def test_inplace_update(instruction_set=instruction_set):
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


def test_vectorization_fixed_size(instruction_set=instruction_set):
    instructions = get_vector_instruction_set(instruction_set=instruction_set)
    configurations = []
    # Fixed size - multiple of four
    arr = np.ones((20 + 2, 24 + 2)) * 5.0
    f, g = ps.fields(f=arr, g=arr)
    configurations.append((arr, f, g))
    # Fixed size - no multiple of four
    arr = np.ones((21 + 2, 25 + 2)) * 5.0
    f, g = ps.fields(f=arr, g=arr)
    configurations.append((arr, f, g))
    # Fixed size - different remainder
    arr = np.ones((23 + 2, 17 + 2)) * 5.0
    f, g = ps.fields(f=arr, g=arr)
    configurations.append((arr, f, g))

    for arr, f, g in configurations:
        update_rule = [ps.Assignment(g[0, 0], f[0, 0] + f[-1, 0] + f[1, 0] + f[0, 1] + f[0, -1] + 42.0)]

        ast = ps.create_kernel(update_rule)
        vectorize(ast, instruction_set=instruction_set)
        code = ps.get_code_str(ast)
        add_instruction = instructions["+"][:instructions["+"].find("(")]
        assert add_instruction in code
        # print(code)

        func = ast.compile()
        dst = np.zeros_like(arr)
        func(g=dst, f=arr)
        np.testing.assert_equal(dst[1:-1, 1:-1], 5 * 5.0 + 42.0)


def test_vectorization_variable_size(instruction_set=instruction_set):
    f, g = ps.fields("f, g : double[2D]")
    update_rule = [ps.Assignment(g[0, 0], f[0, 0] + f[-1, 0] + f[1, 0] + f[0, 1] + f[0, -1] + 42.0)]
    ast = ps.create_kernel(update_rule)

    replace_inner_stride_with_one(ast)
    vectorize(ast, instruction_set=instruction_set)
    func = ast.compile()

    arr = np.ones((23 + 2, 17 + 2)) * 5.0
    dst = np.zeros_like(arr)

    func(g=dst, f=arr)
    np.testing.assert_equal(dst[1:-1, 1:-1], 5 * 5.0 + 42.0)


def test_piecewise1(instruction_set=instruction_set):
    a, b, c, d, e = sp.symbols("a b c d e")
    arr = np.ones((2 ** 3 + 2, 2 ** 4 + 2)) * 5.0

    f, g = ps.fields(f=arr, g=arr)
    update_rule = [ps.Assignment(a, f[1, 0]),
                   ps.Assignment(b, a),
                   ps.Assignment(c, f[0, 0] > 0.0),
                   ps.Assignment(g[0, 0], sp.Piecewise((b + 3 + f[0, 1], c), (0.0, True)))]

    ast = ps.create_kernel(update_rule)
    vectorize(ast, instruction_set=instruction_set)
    func = ast.compile()
    dst = np.zeros_like(arr)
    func(g=dst, f=arr)
    np.testing.assert_equal(dst[1:-1, 1:-1], 5 + 3 + 5.0)


def test_piecewise2(instruction_set=instruction_set):
    arr = np.zeros((20, 20))

    @ps.kernel
    def test_kernel(s):
        f, g = ps.fields(f=arr, g=arr)

        s.condition @= f[0, 0] > 1
        s.result    @= 0.0 if s.condition else 1.0
        g[0, 0]     @= s.result

    ast = ps.create_kernel(test_kernel)
    # ps.show_code(ast)
    vectorize(ast, instruction_set=instruction_set)
    # ps.show_code(ast)
    func = ast.compile()
    func(f=arr, g=arr)
    np.testing.assert_equal(arr, np.ones_like(arr))


def test_piecewise3(instruction_set=instruction_set):
    arr = np.zeros((22, 22))

    @ps.kernel
    def test_kernel(s):
        f, g = ps.fields(f=arr, g=arr)
        s.b     @= f[0, 1]
        g[0, 0] @= 1.0 / (s.b + s.k) if f[0, 0] > 0.0 else 1.0

    ast = ps.create_kernel(test_kernel)
    # ps.show_code(ast)
    vectorize(ast, instruction_set=instruction_set)
    # ps.show_code(ast)
    ast.compile()


def test_logical_operators(instruction_set=instruction_set):
    arr = np.zeros((22, 22))

    @ps.kernel
    def kernel_and(s):
        f, g = ps.fields(f=arr, g=arr)
        s.c @= sp.And(f[0, 1] < 0.0, f[1, 0] < 0.0)
        g[0, 0] @= sp.Piecewise([1.0 / f[1, 0], s.c], [1.0, True])

    ast = ps.create_kernel(kernel_and)
    vectorize(ast, instruction_set=instruction_set)
    ast.compile()

    @ps.kernel
    def kernel_or(s):
        f, g = ps.fields(f=arr, g=arr)
        s.c @= sp.Or(f[0, 1] < 0.0, f[1, 0] < 0.0)
        g[0, 0] @= sp.Piecewise([1.0 / f[1, 0], s.c], [1.0, True])

    ast = ps.create_kernel(kernel_or)
    vectorize(ast, instruction_set=instruction_set)
    ast.compile()

    @ps.kernel
    def kernel_equal(s):
        f, g = ps.fields(f=arr, g=arr)
        s.c @= sp.Eq(f[0, 1], 2.0)
        g[0, 0] @= sp.Piecewise([1.0 / f[1, 0], s.c], [1.0, True])

    ast = ps.create_kernel(kernel_equal)
    vectorize(ast, instruction_set=instruction_set)
    ast.compile()


def test_hardware_query():
    assert {'sse', 'neon', 'sve', 'vsx', 'rvv'}.intersection(supported_instruction_sets)


def test_vectorised_pow(instruction_set=instruction_set):
    arr = np.zeros((24, 24))
    f, g = ps.fields(f=arr, g=arr)

    as1 = ps.Assignment(g[0, 0], sp.Pow(f[0, 0], 2))
    as2 = ps.Assignment(g[0, 0], sp.Pow(f[0, 0], 0.5))
    as3 = ps.Assignment(g[0, 0], sp.Pow(f[0, 0], -0.5))
    as4 = ps.Assignment(g[0, 0], sp.Pow(f[0, 0], 4))
    as5 = ps.Assignment(g[0, 0], sp.Pow(f[0, 0], -4))
    as6 = ps.Assignment(g[0, 0], sp.Pow(f[0, 0], -1))

    ast = ps.create_kernel(as1)
    vectorize(ast, instruction_set=instruction_set)
    ast.compile()

    ast = ps.create_kernel(as2)
    vectorize(ast, instruction_set=instruction_set)
    ast.compile()

    ast = ps.create_kernel(as3)
    vectorize(ast, instruction_set=instruction_set)
    ast.compile()

    ast = ps.create_kernel(as4)
    vectorize(ast, instruction_set=instruction_set)
    ast.compile()

    ast = ps.create_kernel(as5)
    vectorize(ast, instruction_set=instruction_set)
    ast.compile()

    ast = ps.create_kernel(as6)
    vectorize(ast, instruction_set=instruction_set)
    ast.compile()


def test_issue40(*_):
    """https://i10git.cs.fau.de/pycodegen/pystencils/-/issues/40"""
    opt = {'instruction_set': "avx512", 'assume_aligned': False,
           'nontemporal': False, 'assume_inner_stride_one': True}

    src = ps.fields("src(1): double[2D]", layout='fzyx')
    eq = [ps.Assignment(sp.Symbol('rho'), 1.0),
          ps.Assignment(src[0, 0](0), sp.Rational(4, 9) * sp.Symbol('rho'))]

    config = pystencils.config.CreateKernelConfig(target=Target.CPU, cpu_vectorize_info=opt, data_type='float64')
    ast = ps.create_kernel(eq, config=config)

    code = ps.get_code_str(ast)
    assert 'epi32' not in code
