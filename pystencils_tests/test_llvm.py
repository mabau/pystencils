
import pytest
try:
    from pystencils.llvm.llvmjit import generate_and_jit
    from pystencils.llvm import create_kernel, make_python_function
    from pystencils.cpu.cpujit import get_llc_command
    from pystencils import Assignment, Field, Target
    import numpy as np
    import sympy as sp
except ModuleNotFoundError:
    pytest.importorskip("llvmlite")


def test_jacobi_fixed_field_size():
    size = (30, 20)

    src_field_llvm = np.random.rand(*size)
    src_field_py = np.copy(src_field_llvm)
    dst_field_llvm = np.zeros(size)
    dst_field_py = np.zeros(size)

    f = Field.create_from_numpy_array("f", src_field_llvm)
    d = Field.create_from_numpy_array("d", dst_field_llvm)

    jacobi = Assignment(d[0, 0], (f[1, 0] + f[-1, 0] + f[0, 1] + f[0, -1]) / 4)
    ast = create_kernel([jacobi])

    for x in range(1, size[0] - 1):
        for y in range(1, size[1] - 1):
            dst_field_py[x, y] = 0.25 * (src_field_py[x - 1, y] + src_field_py[x + 1, y] +
                                         src_field_py[x, y - 1] + src_field_py[x, y + 1])

    jit = generate_and_jit(ast)
    jit('kernel', dst_field_llvm, src_field_llvm)
    error = np.sum(np.abs(dst_field_py - dst_field_llvm))
    np.testing.assert_almost_equal(error, 0.0)


@pytest.mark.skipif(not get_llc_command(), reason="Tests requires llc in $PATH")
def test_jacobi_fixed_field_size_gpu():
    pytest.importorskip("pycuda")
    size = (30, 20)

    import pycuda.autoinit  # noqa
    from pycuda.gpuarray import to_gpu

    src_field_llvm = np.random.rand(*size)
    src_field_py = np.copy(src_field_llvm)
    dst_field_llvm = np.zeros(size)
    dst_field_py = np.zeros(size)

    f = Field.create_from_numpy_array("f", src_field_py)
    d = Field.create_from_numpy_array("d", dst_field_py)

    src_field_llvm = to_gpu(src_field_llvm)
    dst_field_llvm = to_gpu(dst_field_llvm)

    jacobi = Assignment(d[0, 0], (f[1, 0] + f[-1, 0] + f[0, 1] + f[0, -1]) / 4)
    ast = create_kernel([jacobi], target=Target.GPU)

    for x in range(1, size[0] - 1):
        for y in range(1, size[1] - 1):
            dst_field_py[x, y] = 0.25 * (src_field_py[x - 1, y] + src_field_py[x + 1, y] +
                                         src_field_py[x, y - 1] + src_field_py[x, y + 1])

    jit = generate_and_jit(ast)
    jit('kernel', dst_field_llvm, src_field_llvm)
    error = np.sum(np.abs(dst_field_py - dst_field_llvm.get()))
    np.testing.assert_almost_equal(error, 0.0)


def test_jacobi_variable_field_size():
    size = (3, 3, 3)
    f = Field.create_generic("f", 3)
    d = Field.create_generic("d", 3)
    jacobi = Assignment(d[0, 0, 0], (f[1, 0, 0] + f[-1, 0, 0] + f[0, 1, 0] + f[0, -1, 0]) / 4)
    ast = create_kernel([jacobi])

    src_field_llvm = np.random.rand(*size)
    src_field_py = np.copy(src_field_llvm)
    dst_field_llvm = np.zeros(size)
    dst_field_py = np.zeros(size)

    for x in range(1, size[0] - 1):
        for y in range(1, size[1] - 1):
            for z in range(1, size[2] - 1):
                dst_field_py[x, y, z] = 0.25 * (src_field_py[x - 1, y, z] + src_field_py[x + 1, y, z] +
                                                src_field_py[x, y - 1, z] + src_field_py[x, y + 1, z])

    kernel = make_python_function(ast, {'f': src_field_llvm, 'd': dst_field_llvm})
    kernel()
    error = np.sum(np.abs(dst_field_py - dst_field_llvm))
    np.testing.assert_almost_equal(error, 0.0)


def test_pow_llvm():
    size = (30, 20)

    src_field_llvm = 4 * np.ones(size)
    dst_field_llvm = np.zeros(size)

    f = Field.create_from_numpy_array("f", src_field_llvm)
    d = Field.create_from_numpy_array("d", dst_field_llvm)

    ur = Assignment(d[0, 0], sp.Pow(f[0, 0], -1.0))
    ast = create_kernel([ur])

    jit = generate_and_jit(ast)
    jit('kernel', dst_field_llvm, src_field_llvm)
    assert np.all(0.25 == dst_field_llvm)

    ur = Assignment(d[0, 0], sp.Pow(f[0, 0], 0.5))
    ast = create_kernel([ur])

    jit = generate_and_jit(ast)
    jit('kernel', dst_field_llvm, src_field_llvm)
    assert np.all(2.0 == dst_field_llvm)

    ur = Assignment(d[0, 0], sp.Pow(f[0, 0], 2.0))
    ast = create_kernel([ur])

    jit = generate_and_jit(ast)
    jit('kernel', dst_field_llvm, src_field_llvm)
    assert np.all(16.0 == dst_field_llvm)

    ur = Assignment(d[0, 0], sp.Pow(f[0, 0], 3.0))
    ast = create_kernel([ur])

    jit = generate_and_jit(ast)
    jit('kernel', dst_field_llvm, src_field_llvm)
    assert np.all(64.0 == dst_field_llvm)

    ur = Assignment(d[0, 0], sp.Pow(f[0, 0], 4.0))
    ast = create_kernel([ur])

    jit = generate_and_jit(ast)
    jit('kernel', dst_field_llvm, src_field_llvm)
    assert np.all(256.0 == dst_field_llvm)


def test_piecewise_llvm():
    size = (30, 20)

    src_field_llvm = np.zeros(size)
    dst_field_llvm = np.zeros(size)

    src_field_llvm[0:15, :] = 10.0

    f = Field.create_from_numpy_array("f", src_field_llvm)
    d = Field.create_from_numpy_array("d", dst_field_llvm)

    picewise_test_strict_less_than = Assignment(d[0, 0], sp.Piecewise((1.0, f[0, 0] > 10), (0.0, True)))
    ast = create_kernel([picewise_test_strict_less_than])

    jit = generate_and_jit(ast)
    jit('kernel', dst_field_llvm, src_field_llvm)

    assert (np.all(dst_field_llvm[:, :] == 0.0))

    src_field_llvm = np.zeros(size)
    dst_field_llvm = np.zeros(size)

    src_field_llvm[0:15, :] = 10.0

    picewise_test_less_than = Assignment(d[0, 0], sp.Piecewise((1.0, f[0, 0] >= 10), (0.0, True)))
    ast = create_kernel([picewise_test_less_than])

    jit = generate_and_jit(ast)
    jit('kernel', dst_field_llvm, src_field_llvm)

    assert (np.all(dst_field_llvm[0:15, :] == 1.0))

    src_field_llvm = np.zeros(size)
    dst_field_llvm = np.zeros(size)

    src_field_llvm[0:15, :] = 10.0

    picewise_test_strict_greater_than = Assignment(d[0, 0], sp.Piecewise((1.0, f[0, 0] < 5), (0.0, True)))
    ast = create_kernel([picewise_test_strict_greater_than])

    jit = generate_and_jit(ast)
    jit('kernel', dst_field_llvm, src_field_llvm)

    assert (np.all(dst_field_llvm[15:, :] == 1.0))

    src_field_llvm = np.zeros(size)
    dst_field_llvm = np.zeros(size)

    src_field_llvm[0:15, :] = 10.0

    picewise_test_greater_than = Assignment(d[0, 0], sp.Piecewise((1.0, f[0, 0] <= 10), (0.0, True)))
    ast = create_kernel([picewise_test_greater_than])

    jit = generate_and_jit(ast)
    jit('kernel', dst_field_llvm, src_field_llvm)

    assert (np.all(dst_field_llvm[:, :] == 1.0))

    src_field_llvm = np.zeros(size)
    dst_field_llvm = np.zeros(size)

    src_field_llvm[0:15, :] = 10.0

    picewise_test_equality = Assignment(d[0, 0], sp.Piecewise((1.0, sp.Equality(f[0, 0], 10.0)), (0.0, True)))
    ast = create_kernel([picewise_test_equality])

    jit = generate_and_jit(ast)
    jit('kernel', dst_field_llvm, src_field_llvm)

    assert (np.all(dst_field_llvm[0:15, :] == 1.0))

    src_field_llvm = np.zeros(size)
    dst_field_llvm = np.zeros(size)

    src_field_llvm[0:15, :] = 10.0

    picewise_test_unequality = Assignment(d[0, 0], sp.Piecewise((1.0, sp.Unequality(f[0, 0], 10.0)), (0.0, True)))
    ast = create_kernel([picewise_test_unequality])

    jit = generate_and_jit(ast)
    jit('kernel', dst_field_llvm, src_field_llvm)

    assert (np.all(dst_field_llvm[15:, :] == 1.0))


def test_piecewise_or_llvm():
    size = (30, 20)

    src_field_llvm = np.zeros(size)
    dst_field_llvm = np.zeros(size)

    src_field_llvm[0:15, :] = 10.5

    f = Field.create_from_numpy_array("f", src_field_llvm)
    d = Field.create_from_numpy_array("d", dst_field_llvm)

    picewise_test_or = Assignment(d[0, 0], sp.Piecewise((1.0, sp.Or(f[0, 0] > 11, f[0, 0] < 10)), (0.0, True)))
    ast = create_kernel([picewise_test_or])

    jit = generate_and_jit(ast)
    jit('kernel', dst_field_llvm, src_field_llvm)

    assert (np.all(dst_field_llvm[0:15, :] == 0.0))


def test_print_function_llvm():
    size = (30, 20)

    src_field_llvm = np.zeros(size)
    dst_field_llvm = np.zeros(size)

    src_field_llvm[0:15, :] = 0.0

    f = Field.create_from_numpy_array("f", src_field_llvm)
    d = Field.create_from_numpy_array("d", dst_field_llvm)

    up = Assignment(d[0, 0], sp.sin(f[0, 0]))
    ast = create_kernel([up])

    # kernel = make_python_function(ast, {'f': src_field_llvm, 'd': dst_field_llvm})
    jit = generate_and_jit(ast)
    jit('kernel', dst_field_llvm, src_field_llvm)

    assert (np.all(dst_field_llvm[:, :] == 0.0))


if __name__ == "__main__":
    test_jacobi_fixed_field_size_gpu()
