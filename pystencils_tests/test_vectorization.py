import numpy as np
import sympy as sp

import pystencils as ps
from pystencils.backends.simd_instruction_sets import get_supported_instruction_sets
from pystencils.cpu.vectorization import vectorize
from pystencils.transformations import replace_inner_stride_with_one


def test_vector_type_propagation():
    a, b, c, d, e = sp.symbols("a b c d e")
    arr = np.ones((2 ** 2 + 2, 2 ** 3 + 2))
    arr *= 10.0

    f, g = ps.fields(f=arr, g=arr)
    update_rule = [ps.Assignment(a, f[1, 0]),
                   ps.Assignment(b, a),
                   ps.Assignment(g[0, 0], b + 3 + f[0, 1])]

    ast = ps.create_kernel(update_rule)
    vectorize(ast)

    func = ast.compile()
    dst = np.zeros_like(arr)
    func(g=dst, f=arr)
    np.testing.assert_equal(dst[1:-1, 1:-1], 2 * 10.0 + 3)


def test_inplace_update():
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

    ast = ps.create_kernel(update_rule, cpu_vectorize_info={'instruction_set': 'avx'})
    kernel = ast.compile()
    kernel(f=arr)
    np.testing.assert_equal(arr, 2)

def test_vectorization_fixed_size():
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
        vectorize(ast)

        func = ast.compile()
        dst = np.zeros_like(arr)
        func(g=dst, f=arr)
        np.testing.assert_equal(dst[1:-1, 1:-1], 5 * 5.0 + 42.0)


def test_vectorization_variable_size():
    f, g = ps.fields("f, g : double[2D]")
    update_rule = [ps.Assignment(g[0, 0], f[0, 0] + f[-1, 0] + f[1, 0] + f[0, 1] + f[0, -1] + 42.0)]
    ast = ps.create_kernel(update_rule)

    replace_inner_stride_with_one(ast)
    vectorize(ast)
    func = ast.compile()

    arr = np.ones((23 + 2, 17 + 2)) * 5.0
    dst = np.zeros_like(arr)

    func(g=dst, f=arr)
    np.testing.assert_equal(dst[1:-1, 1:-1], 5 * 5.0 + 42.0)


def test_piecewise1():
    a, b, c, d, e = sp.symbols("a b c d e")
    arr = np.ones((2 ** 3 + 2, 2 ** 4 + 2)) * 5.0

    f, g = ps.fields(f=arr, g=arr)
    update_rule = [ps.Assignment(a, f[1, 0]),
                   ps.Assignment(b, a),
                   ps.Assignment(c, f[0, 0] > 0.0),
                   ps.Assignment(g[0, 0], sp.Piecewise((b + 3 + f[0, 1], c), (0.0, True)))]

    ast = ps.create_kernel(update_rule)
    vectorize(ast)
    func = ast.compile()
    dst = np.zeros_like(arr)
    func(g=dst, f=arr)
    np.testing.assert_equal(dst[1:-1, 1:-1], 5 + 3 + 5.0)


def test_piecewise2():

    arr = np.zeros((20, 20))

    @ps.kernel
    def test_kernel(s):
        f, g = ps.fields(f=arr, g=arr)

        s.condition @= f[0, 0] > 1
        s.result    @= 0.0 if s.condition else 1.0
        g[0, 0]     @= s.result

    ast = ps.create_kernel(test_kernel)
    vectorize(ast)
    func = ast.compile()
    func(f=arr, g=arr)
    np.testing.assert_equal(arr, np.ones_like(arr))


def test_piecewise3():

    arr = np.zeros((22, 22))

    @ps.kernel
    def test_kernel(s):
        f, g = ps.fields(f=arr, g=arr)
        s.b     @= f[0, 1]
        g[0, 0] @= 1.0 / (s.b + s.k) if f[0, 0] > 0.0 else 1.0

    ast = ps.create_kernel(test_kernel)
    vectorize(ast)
    ast.compile()


def test_logical_operators():
    arr = np.zeros((22, 22))

    @ps.kernel
    def test_kernel(s):
        f, g = ps.fields(f=arr, g=arr)
        s.c @= sp.And(f[0, 1] < 0.0, f[1, 0] < 0.0)
        g[0, 0] @= sp.Piecewise([1.0 / f[1, 0], s.c], [1.0, True])

    ast = ps.create_kernel(test_kernel)
    vectorize(ast)
    ast.compile()


def test_hardware_query():
    instruction_sets = get_supported_instruction_sets()
    assert 'sse' in instruction_sets
