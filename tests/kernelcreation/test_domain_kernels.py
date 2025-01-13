import pytest
import sympy as sp
import numpy as np

from dataclasses import replace

from pystencils import (
    fields,
    Field,
    AssignmentCollection,
    Target,
    CreateKernelConfig,
)
from pystencils.assignment import assignment_from_stencil

from pystencils import create_kernel, Kernel
from pystencils.backend.emission import emit_code


def inspect_dp_kernel(kernel: Kernel, gen_config: CreateKernelConfig):
    code = emit_code(kernel)

    match gen_config.target:
        case Target.X86_SSE:
            assert "_mm_loadu_pd" in code
            assert "_mm_storeu_pd" in code
        case Target.X86_AVX:
            assert "_mm256_loadu_pd" in code
            assert "_mm256_storeu_pd" in code
        case Target.X86_AVX512:
            assert "_mm512_loadu_pd" in code
            assert "_mm512_storeu_pd" in code


def test_filter_kernel(gen_config):
    if gen_config.target == Target.CUDA:
        import cupy as cp

        xp = cp
    else:
        xp = np

    weight = sp.Symbol("weight")
    stencil = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    src, dst = fields("src, dst: [2D]")
    asm = assignment_from_stencil(stencil, src, dst, normalization_factor=weight)
    asms = AssignmentCollection([asm])

    ast = create_kernel(asms, gen_config)
    inspect_dp_kernel(ast, gen_config)
    kernel = ast.compile()

    src_arr = xp.ones((42, 31))
    dst_arr = xp.zeros_like(src_arr)

    kernel(src=src_arr, dst=dst_arr, weight=2.0)

    expected = xp.zeros_like(src_arr)
    expected[1:-1, 1:-1].fill(18.0)

    xp.testing.assert_allclose(dst_arr, expected)


def test_filter_kernel_fixedsize(gen_config):
    if gen_config.target == Target.CUDA:
        import cupy as cp

        xp = cp
    else:
        xp = np

    weight = sp.Symbol("weight")
    stencil = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    src_arr = xp.ones((42, 31))
    dst_arr = xp.zeros_like(src_arr)

    src = Field.create_from_numpy_array("src", src_arr)
    dst = Field.create_from_numpy_array("dst", dst_arr)

    asm = assignment_from_stencil(stencil, src, dst, normalization_factor=weight)
    asms = AssignmentCollection([asm])

    ast = create_kernel(asms, gen_config)
    inspect_dp_kernel(ast, gen_config)
    kernel = ast.compile()

    kernel(src=src_arr, dst=dst_arr, weight=2.0)

    expected = xp.zeros_like(src_arr)
    expected[1:-1, 1:-1].fill(18.0)

    xp.testing.assert_allclose(dst_arr, expected)
