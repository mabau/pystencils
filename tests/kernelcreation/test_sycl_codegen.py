"""
Since we don't have a JIT compiler for SYCL, these tests can only
perform dry-dock testing.
If the SYCL target should ever become non-experimental, we need to
find a way to properly test SYCL kernels in execution.

These tests primarily check that the code generation driver runs
successfully for the SYCL target.
"""

import sympy as sp
from pystencils import (
    create_kernel,
    Target,
    fields,
    Assignment,
    CreateKernelConfig,
)


def test_sycl_kernel_static():
    src, dst = fields("src, dst: [2D]")
    asm = Assignment(dst.center(), sp.sin(src.center()) + sp.cos(src.center()))

    cfg = CreateKernelConfig(target=Target.SYCL)
    kernel = create_kernel(asm, cfg)

    code_string = kernel.get_c_code()

    assert "sycl::id< 2 >" in code_string
    assert "sycl::sin(" in code_string
    assert "sycl::cos(" in code_string


def test_sycl_kernel_manual_block_size():
    src, dst = fields("src, dst: [2D]")
    asm = Assignment(dst.center(), sp.sin(src.center()) + sp.cos(src.center()))

    cfg = CreateKernelConfig(target=Target.SYCL)
    cfg.sycl.automatic_block_size = False
    kernel = create_kernel(asm, cfg)

    code_string = kernel.get_c_code()

    assert "sycl::nd_item< 2 >" in code_string
