import pytest
import sympy as sp
import numpy as np

from pystencils import fields, Field, AssignmentCollection, Target, CreateKernelConfig
from pystencils.assignment import assignment_from_stencil

from pystencils.kernelcreation import create_kernel


@pytest.mark.parametrize("target", (Target.GenericCPU, Target.CUDA))
def test_filter_kernel(target):
    if target == Target.CUDA:
        xp = pytest.importorskip("cupy")
    else:
        xp = np

    weight = sp.Symbol("weight")
    stencil = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]

    src, dst = fields("src, dst: [2D]")
    asm = assignment_from_stencil(stencil, src, dst, normalization_factor=weight)
    asms = AssignmentCollection([asm])

    gen_config = CreateKernelConfig(target=target)
    ast = create_kernel(asms, gen_config)
    kernel = ast.compile()

    src_arr = xp.ones((42, 31))
    dst_arr = xp.zeros_like(src_arr)

    kernel(src=src_arr, dst=dst_arr, weight=2.0)

    expected = xp.zeros_like(src_arr)
    expected[1:-1, 1:-1].fill(18.0)

    xp.testing.assert_allclose(dst_arr, expected)


@pytest.mark.parametrize("target", (Target.GenericCPU, Target.CUDA))
def test_filter_kernel_fixedsize(target):
    if target == Target.CUDA:
        xp = pytest.importorskip("cupy")
    else:
        xp = np

    weight = sp.Symbol("weight")
    stencil = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]

    src_arr = xp.ones((42, 31))
    dst_arr = xp.zeros_like(src_arr)

    src = Field.create_from_numpy_array("src", src_arr)
    dst = Field.create_from_numpy_array("dst", dst_arr)
    
    asm = assignment_from_stencil(stencil, src, dst, normalization_factor=weight)
    asms = AssignmentCollection([asm])

    gen_config = CreateKernelConfig(target=target)
    ast = create_kernel(asms, gen_config)
    kernel = ast.compile()

    kernel(src=src_arr, dst=dst_arr, weight=2.0)

    expected = xp.zeros_like(src_arr)
    expected[1:-1, 1:-1].fill(18.0)

    xp.testing.assert_allclose(dst_arr, expected)
