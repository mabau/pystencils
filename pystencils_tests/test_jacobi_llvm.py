import numpy as np

from pystencils import Assignment, Field, show_code
from pystencils.llvm import create_kernel, make_python_function
from pystencils.llvm.llvmjit import generate_and_jit


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


def test_jacobi_fixed_field_size_gpu():
    size = (30, 20)

    src_field_llvm = np.random.rand(*size)
    src_field_py = np.copy(src_field_llvm)
    dst_field_llvm = np.zeros(size)
    dst_field_py = np.zeros(size)

    f = Field.create_from_numpy_array("f", src_field_llvm)
    d = Field.create_from_numpy_array("d", dst_field_llvm)

    jacobi = Assignment(d[0, 0], (f[1, 0] + f[-1, 0] + f[0, 1] + f[0, -1]) / 4)
    ast = create_kernel([jacobi], target='gpu')
    print(show_code(ast))

    for x in range(1, size[0] - 1):
        for y in range(1, size[1] - 1):
            dst_field_py[x, y] = 0.25 * (src_field_py[x - 1, y] + src_field_py[x + 1, y] +
                                         src_field_py[x, y - 1] + src_field_py[x, y + 1])

    jit = generate_and_jit(ast)
    jit('kernel', dst_field_llvm, src_field_llvm)
    error = np.sum(np.abs(dst_field_py - dst_field_llvm))
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


if __name__ == "__main__":
    test_jacobi_fixed_field_size_gpu()
