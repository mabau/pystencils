import sympy as sp
import numpy as np

from pystencils import fields, Field, AssignmentCollection
from pystencils.sympyextensions.assignmentcollection.assignment import assignment_from_stencil

from pystencils.nbackend.kernelcreation import create_kernel


def test_filter_kernel():
    weight = sp.Symbol("weight")
    stencil = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]

    src, dst = fields("src, dst: [2D]")
    asm = assignment_from_stencil(stencil, src, dst, normalization_factor=weight)
    asms = AssignmentCollection([asm])

    ast = create_kernel(asms)
    kernel = ast.compile()

    src_arr = np.ones((42, 42))
    dst_arr = np.zeros_like(src_arr)

    kernel(src=src_arr, dst=dst_arr, weight=2.0)

    expected = np.zeros_like(src_arr)
    expected[1:-1, 1:-1].fill(18.0)

    np.testing.assert_allclose(dst_arr, expected)


def test_filter_kernel_fixedsize():
    weight = sp.Symbol("weight")
    stencil = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]

    src_arr = np.ones((42, 42))
    dst_arr = np.zeros_like(src_arr)

    src = Field.create_from_numpy_array("src", src_arr)
    dst = Field.create_from_numpy_array("dst", dst_arr)
    
    asm = assignment_from_stencil(stencil, src, dst, normalization_factor=weight)
    asms = AssignmentCollection([asm])

    ast = create_kernel(asms)
    kernel = ast.compile()

    kernel(src=src_arr, dst=dst_arr, weight=2.0)

    expected = np.zeros_like(src_arr)
    expected[1:-1, 1:-1].fill(18.0)

    np.testing.assert_allclose(dst_arr, expected)
