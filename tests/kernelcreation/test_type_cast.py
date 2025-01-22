import numpy as np
import pytest
from itertools import product

from pystencils import (
    create_kernel,
    Target,
    Assignment,
    Field,
)
from pystencils.sympyextensions.typed_sympy import CastFunc


AVAIL_TARGETS_NO_SSE = [t for t in Target.available_targets() if Target._SSE not in t]


target_and_dtype = pytest.mark.parametrize(
    "target, from_type, to_type",
    list(
        product(
            [
                t
                for t in AVAIL_TARGETS_NO_SSE
                if Target._X86 in t and Target._AVX512 not in t
            ],
            [np.int32, np.float32, np.float64],
            [np.int32, np.float32, np.float64],
        )
    )
    + list(
        product(
            [
                t
                for t in AVAIL_TARGETS_NO_SSE
                if Target._X86 not in t or Target._AVX512 in t
            ],
            [np.int32, np.int64, np.float32, np.float64],
            [np.int32, np.int64, np.float32, np.float64],
        )
    ),
)


@target_and_dtype
def test_type_cast(gen_config, xp, from_type, to_type):
    if np.issubdtype(from_type, np.floating):
        inp = xp.array([-1.25, -0, 1.5, 3, -5, -312, 42, 6.625, -9], dtype=from_type)
    else:
        inp = xp.array([-1, 0, 1, 3, -5, -312, 42, 6, -9], dtype=from_type)

    outp = xp.zeros_like(inp).astype(to_type)
    truncated = inp.astype(to_type)
    rounded = xp.round(inp).astype(to_type)

    inp_field = Field.create_from_numpy_array("inp", inp)
    outp_field = Field.create_from_numpy_array("outp", outp)

    asms = [Assignment(outp_field.center(), CastFunc(inp_field.center(), to_type))]

    kernel = create_kernel(asms, gen_config)
    kfunc = kernel.compile()
    kfunc(inp=inp, outp=outp)

    if np.issubdtype(from_type, np.floating) and not np.issubdtype(
        to_type, np.floating
    ):
        # rounding mode depends on platform
        try:
            xp.testing.assert_array_equal(outp, truncated)
        except AssertionError:
            xp.testing.assert_array_equal(outp, rounded)
    else:
        xp.testing.assert_array_equal(outp, truncated)
