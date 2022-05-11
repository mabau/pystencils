import numpy as np
import sympy as sp

import pystencils as ps


def jacobi(dst, src):
    assert dst.spatial_dimensions == src.spatial_dimensions
    assert src.index_dimensions == 0 and dst.index_dimensions == 0
    neighbors = []
    for d in range(src.spatial_dimensions):
        neighbors += [src.neighbor(d, offset) for offset in (1, -1)]
    return ps.Assignment(dst.center, sp.Add(*neighbors) / len(neighbors))


def check_equivalence(assignments, src_arr):
    for openmp in (False, True):
        for vectorization in [False, {'assume_inner_stride_one': True}]:
            with_blocking = ps.create_kernel(assignments, cpu_blocking=(8, 16, 4), cpu_openmp=openmp,
                                             cpu_vectorize_info=vectorization).compile()
            with_blocking_only_over_y = ps.create_kernel(assignments, cpu_blocking=(0, 16, 0), cpu_openmp=openmp,
                                                         cpu_vectorize_info=vectorization).compile()
            without_blocking = ps.create_kernel(assignments).compile()

            only_omp = ps.create_kernel(assignments, cpu_openmp=2).compile()

            print(f"  openmp {openmp}, vectorization {vectorization}")
            dst_arr = np.zeros_like(src_arr)
            dst2_arr = np.zeros_like(src_arr)
            dst3_arr = np.zeros_like(src_arr)
            ref_arr = np.zeros_like(src_arr)
            np.copyto(src_arr, np.random.rand(*src_arr.shape))
            with_blocking(src=src_arr, dst=dst_arr)
            with_blocking_only_over_y(src=src_arr, dst=dst2_arr)
            without_blocking(src=src_arr, dst=ref_arr)
            only_omp(src=src_arr, dst=dst3_arr)
            np.testing.assert_almost_equal(ref_arr, dst_arr)
            np.testing.assert_almost_equal(ref_arr, dst2_arr)
            np.testing.assert_almost_equal(ref_arr, dst3_arr)


def test_jacobi3d_var_size():
    src, dst = ps.fields("src, dst: double[3D]", layout='c')

    print("Var Size: Smaller than block sizes")
    arr = np.empty([4, 5, 6])
    check_equivalence(jacobi(dst, src), arr)

    print("Var Size: Large non divisible sizes")
    arr = np.empty([100, 80, 9])
    check_equivalence(jacobi(dst, src), arr)

    print("Var Size: Multiples of block sizes")
    arr = np.empty([8*4, 16*2, 4*3])
    check_equivalence(jacobi(dst, src), arr)


def test_jacobi3d_fixed_size():
    print("Fixed Size: Large non divisible sizes")
    arr = np.empty([10, 10, 9])
    src, dst = ps.fields("src, dst: double[3D]", src=arr, dst=arr)
    check_equivalence(jacobi(dst, src), arr)

    print("Fixed Size: Smaller than block sizes")
    arr = np.empty([4, 5, 6])
    src, dst = ps.fields("src, dst: double[3D]", src=arr, dst=arr)
    check_equivalence(jacobi(dst, src), arr)

    print("Fixed Size: Multiples of block sizes")
    arr = np.empty([8*4, 16*2, 4*3])
    src, dst = ps.fields("src, dst: double[3D]", src=arr, dst=arr)
    check_equivalence(jacobi(dst, src), arr)


def test_jacobi3d_fixed_field_size():
    src, dst = ps.fields("src, dst: double[3, 5, 6]", layout='c')

    print("Fixed Field Size: Smaller than block sizes")
    arr = np.empty([3, 5, 6])
    check_equivalence(jacobi(dst, src), arr)
