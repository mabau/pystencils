import pytest
import sympy as sp
import numpy as np
from dataclasses import dataclass
from itertools import chain

from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    AstFactory,
    FullIterationSpace,
)
from pystencils.backend.platforms import GenericVectorCpu, X86VectorArch, X86VectorCpu
from pystencils.backend.ast.structural import PsBlock
from pystencils.backend.transformations import (
    LoopVectorizer,
    SelectIntrinsics,
    LowerToC,
)
from pystencils.backend.constants import PsConstant
from pystencils.backend.kernelfunction import create_cpu_kernel_function
from pystencils.backend.jit import LegacyCpuJit

from pystencils import Target, fields, Assignment, Field
from pystencils.field import create_numpy_array_with_layout
from pystencils.types import PsScalarType, PsIntegerType
from pystencils.types.quick import SInt, Fp


@dataclass
class VectorTestSetup:
    platform: GenericVectorCpu
    lanes: int
    numeric_dtype: PsScalarType
    index_dtype: PsIntegerType

    @property
    def name(self) -> str:
        if isinstance(self.platform, X86VectorCpu):
            match self.platform.vector_arch:
                case X86VectorArch.SSE:
                    isa = "SSE"
                case X86VectorArch.AVX:
                    isa = "AVX"
                case X86VectorArch.AVX512:
                    isa = "AVX512"
                case X86VectorArch.AVX512_FP16:
                    isa = "AVX512_FP16"
        else:
            assert False

        return f"{isa}/{self.numeric_dtype}<{self.lanes}>/{self.index_dtype}"


def get_setups(target: Target) -> list[VectorTestSetup]:
    match target:
        case Target.X86_SSE:
            sse_platform = X86VectorCpu(X86VectorArch.SSE)
            return [
                VectorTestSetup(sse_platform, 4, Fp(32), SInt(32)),
                VectorTestSetup(sse_platform, 2, Fp(64), SInt(64)),
            ]

        case Target.X86_AVX:
            avx_platform = X86VectorCpu(X86VectorArch.AVX)
            return [
                VectorTestSetup(avx_platform, 4, Fp(32), SInt(32)),
                VectorTestSetup(avx_platform, 8, Fp(32), SInt(32)),
                VectorTestSetup(avx_platform, 2, Fp(64), SInt(64)),
                VectorTestSetup(avx_platform, 4, Fp(64), SInt(64)),
            ]

        case Target.X86_AVX512:
            avx512_platform = X86VectorCpu(X86VectorArch.AVX512)
            return [
                VectorTestSetup(avx512_platform, 4, Fp(32), SInt(32)),
                VectorTestSetup(avx512_platform, 8, Fp(32), SInt(32)),
                VectorTestSetup(avx512_platform, 16, Fp(32), SInt(32)),
                VectorTestSetup(avx512_platform, 2, Fp(64), SInt(64)),
                VectorTestSetup(avx512_platform, 4, Fp(64), SInt(64)),
                VectorTestSetup(avx512_platform, 8, Fp(64), SInt(64)),
            ]

        case Target.X86_AVX512_FP16:
            avx512_platform = X86VectorCpu(X86VectorArch.AVX512_FP16)
            return [
                VectorTestSetup(avx512_platform, 8, Fp(16), SInt(32)),
                VectorTestSetup(avx512_platform, 16, Fp(16), SInt(32)),
                VectorTestSetup(avx512_platform, 32, Fp(16), SInt(32)),
            ]

        case _:
            return []


TEST_SETUPS: list[VectorTestSetup] = list(
    chain.from_iterable(get_setups(t) for t in Target.available_vector_cpu_targets())
)

TEST_IDS = [t.name for t in TEST_SETUPS]


def create_vector_kernel(
    assignments: list[Assignment],
    field: Field,
    setup: VectorTestSetup,
    ghost_layers: int = 0,
):
    ctx = KernelCreationContext(
        default_dtype=setup.numeric_dtype, index_dtype=setup.index_dtype
    )

    factory = AstFactory(ctx)

    ispace = FullIterationSpace.create_with_ghost_layers(ctx, ghost_layers, field)
    ctx.set_iteration_space(ispace)

    body = PsBlock([factory.parse_sympy(asm) for asm in assignments])

    loop_order = field.layout
    loop_nest = factory.loops_from_ispace(ispace, body, loop_order)

    for field in ctx.fields:
        #   Set inner strides to one to ensure packed memory access
        buf = ctx.get_buffer(field)
        buf.strides[0] = PsConstant(1, ctx.index_dtype)

    vectorize = LoopVectorizer(ctx, setup.lanes)
    loop_nest = vectorize.vectorize_select_loops(
        loop_nest, lambda l: l.counter.symbol.name == "ctr_0"
    )

    select_intrin = SelectIntrinsics(ctx, setup.platform)
    loop_nest = select_intrin(loop_nest)

    lower = LowerToC(ctx)
    loop_nest = lower(loop_nest)

    func = create_cpu_kernel_function(
        ctx,
        setup.platform,
        PsBlock([loop_nest]),
        "vector_kernel",
        Target.CPU,
        LegacyCpuJit(),
    )

    kernel = func.compile()
    return kernel


@pytest.mark.parametrize("setup", TEST_SETUPS, ids=TEST_IDS)
@pytest.mark.parametrize("ghost_layers", [0, 2])
def test_update_kernel(setup: VectorTestSetup, ghost_layers: int):
    src, dst = fields(f"src(2), dst(4): {setup.numeric_dtype}[2D]", layout="fzyx")

    x = sp.symbols("x_:4")

    update = [
        Assignment(x[0], src[0, 0](0) + src[0, 0](1)),
        Assignment(x[1], src[0, 0](0) - src[0, 0](1)),
        Assignment(x[2], src[0, 0](0) * src[0, 0](1)),
        Assignment(x[3], src[0, 0](0) / src[0, 0](1)),
        Assignment(dst.center(0), x[0]),
        Assignment(dst.center(1), x[1]),
        Assignment(dst.center(2), x[2]),
        Assignment(dst.center(3), x[3]),
    ]

    kernel = create_vector_kernel(update, src, setup, ghost_layers)

    shape = (23, 17)

    rgen = np.random.default_rng(seed=1648)
    src_arr = create_numpy_array_with_layout(
        shape + (2,), layout=(2, 1, 0), dtype=setup.numeric_dtype.numpy_dtype
    )
    rgen.random(dtype=setup.numeric_dtype.numpy_dtype, out=src_arr)

    dst_arr = create_numpy_array_with_layout(
        shape + (4,), layout=(2, 1, 0), dtype=setup.numeric_dtype.numpy_dtype
    )
    dst_arr[:] = 0.0

    check_arr = np.zeros_like(dst_arr)
    check_arr[:, :, 0] = src_arr[:, :, 0] + src_arr[:, :, 1]
    check_arr[:, :, 1] = src_arr[:, :, 0] - src_arr[:, :, 1]
    check_arr[:, :, 2] = src_arr[:, :, 0] * src_arr[:, :, 1]
    check_arr[:, :, 3] = src_arr[:, :, 0] / src_arr[:, :, 1]

    kernel(src=src_arr, dst=dst_arr)

    resolution = np.finfo(setup.numeric_dtype.numpy_dtype).resolution
    gls = ghost_layers
    
    np.testing.assert_allclose(
        dst_arr[gls:-gls, gls:-gls, :],
        check_arr[gls:-gls, gls:-gls, :],
        rtol=resolution,
    )

    if gls != 0:
        for i in range(gls):
            np.testing.assert_equal(dst_arr[i, :, :], 0.0)
            np.testing.assert_equal(dst_arr[-i, :, :], 0.0)
            np.testing.assert_equal(dst_arr[:, i, :], 0.0)
            np.testing.assert_equal(dst_arr[:, -i, :], 0.0)


@pytest.mark.parametrize("setup", TEST_SETUPS, ids=TEST_IDS)
def test_trailing_iterations(setup: VectorTestSetup):
    f = fields(f"f(1): {setup.numeric_dtype}[1D]", layout="fzyx")

    update = [Assignment(f(0), 2 * f(0))]

    kernel = create_vector_kernel(update, f, setup)

    for trailing_iters in range(setup.lanes):
        shape = (setup.lanes * 12 + trailing_iters, 1)
        f_arr = create_numpy_array_with_layout(
            shape, layout=(1, 0), dtype=setup.numeric_dtype.numpy_dtype
        )

        f_arr[:] = 1.0

        kernel(f=f_arr)

        np.testing.assert_equal(f_arr, 2.0)
