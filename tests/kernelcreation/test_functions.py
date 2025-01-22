import sympy as sp
import numpy as np
import pytest
from dataclasses import replace
from itertools import product

from pystencils import (
    fields,
    create_kernel,
    CreateKernelConfig,
    Target,
    Assignment,
    Field,
)
from pystencils.backend.ast import dfs_preorder
from pystencils.backend.ast.expressions import PsCall


def unary_function(name, xp):
    return {
        "exp": (sp.exp, xp.exp),
        "log": (sp.log, xp.log),
        "sin": (sp.sin, xp.sin),
        "cos": (sp.cos, xp.cos),
        "tan": (sp.tan, xp.tan),
        "sinh": (sp.sinh, xp.sinh),
        "cosh": (sp.cosh, xp.cosh),
        "asin": (sp.asin, xp.arcsin),
        "acos": (sp.acos, xp.arccos),
        "atan": (sp.atan, xp.arctan),
        "abs": (sp.Abs, xp.abs),
        "floor": (sp.floor, xp.floor),
        "ceil": (sp.ceiling, xp.ceil),
    }[name]


def binary_function(name, xp):
    return {
        "min": (sp.Min, xp.fmin),
        "max": (sp.Max, xp.fmax),
        "pow": (sp.Pow, xp.power),
        "atan2": (sp.atan2, xp.arctan2),
    }[name]


AVAIL_TARGETS = Target.available_targets()


@pytest.mark.parametrize(
    "function_name, target",
    list(
        product(
            (
                "exp",
                "log",
                "sin",
                "cos",
                "tan",
                "sinh",
                "cosh",
                "asin",
                "acos",
                "atan",
            ),
            [t for t in AVAIL_TARGETS if Target._X86 not in t],
        )
    )
    + list(
        product(
            ["floor", "ceil"], [t for t in AVAIL_TARGETS if Target._AVX512 not in t]
        )
    )
    + list(product(["abs"], AVAIL_TARGETS)),
)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_unary_functions(gen_config, xp, function_name, dtype):
    sp_func, xp_func = unary_function(function_name, xp)
    resolution = np.finfo(dtype).resolution

    #   Array size should be larger than eight, such that vectorized kernels don't just run their remainder loop
    inp = xp.array([0.1, 0.2, 0.0, -0.8, -1.6, -12.592, xp.pi, xp.e, -0.3], dtype=dtype)
    outp = xp.zeros_like(inp)

    reference = xp_func(inp)

    inp_field = Field.create_from_numpy_array("inp", inp)
    outp_field = inp_field.new_field_with_different_name("outp")

    asms = [Assignment(outp_field.center(), sp_func(inp_field.center()))]
    gen_config = replace(gen_config, default_dtype=dtype)

    kernel = create_kernel(asms, gen_config)
    kfunc = kernel.compile()
    kfunc(inp=inp, outp=outp)

    xp.testing.assert_allclose(outp, reference, rtol=resolution)


@pytest.mark.parametrize(
    "function_name,target",
    list(product(["min", "max"], AVAIL_TARGETS))
    + list(
        product(["pow", "atan2"], [t for t in AVAIL_TARGETS if Target._X86 not in t])
    ),
)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_binary_functions(gen_config, xp, function_name, dtype):
    sp_func, xp_func = binary_function(function_name, xp)
    resolution: dtype = np.finfo(dtype).resolution

    inp = xp.array([0.1, 0.2, 0.3, -0.8, -1.6, -12.592, xp.pi, xp.e, 0.0], dtype=dtype)
    inp2 = xp.array(
        [3.1, -0.5, 21.409, 11.0, 1.0, -14e3, 2.0 * xp.pi, -xp.e, 0.0],
        dtype=dtype,
    )
    outp = xp.zeros_like(inp)

    reference = xp_func(inp, inp2)

    inp_field = Field.create_from_numpy_array("inp", inp)
    inp2_field = Field.create_from_numpy_array("inp2", inp)
    outp_field = inp_field.new_field_with_different_name("outp")

    asms = [
        Assignment(
            outp_field.center(), sp_func(inp_field.center(), inp2_field.center())
        )
    ]
    gen_config = replace(gen_config, default_dtype=dtype)

    kernel = create_kernel(asms, gen_config)
    kfunc = kernel.compile()
    kfunc(inp=inp, inp2=inp2, outp=outp)

    xp.testing.assert_allclose(outp, reference, rtol=resolution)


dtype_and_target_for_integer_funcs = pytest.mark.parametrize(
    "dtype, target",
    list(product([np.int32], [t for t in AVAIL_TARGETS if t is not Target.CUDA]))
    + list(
        product(
            [np.int64],
            [
                t
                for t in AVAIL_TARGETS
                if t not in (Target.X86_SSE, Target.X86_AVX, Target.CUDA)
            ],
        )
    ),
)


@dtype_and_target_for_integer_funcs
def test_integer_abs(gen_config, xp, dtype):
    sp_func, xp_func = unary_function("abs", xp)

    smallest = np.iinfo(dtype).min
    largest = np.iinfo(dtype).max

    inp = xp.array([-1, 0, 1, 3, -5, -312, smallest + 1, largest], dtype=dtype)

    outp = xp.zeros_like(inp)
    reference = xp_func(inp)

    inp_field = Field.create_from_numpy_array("inp", inp)
    outp_field = inp_field.new_field_with_different_name("outp")

    asms = [Assignment(outp_field.center(), sp_func(inp_field.center()))]
    gen_config = replace(gen_config, default_dtype=dtype)

    kernel = create_kernel(asms, gen_config)
    kfunc = kernel.compile()
    kfunc(inp=inp, outp=outp)

    xp.testing.assert_array_equal(outp, reference)


@pytest.mark.parametrize("function_name", ("min", "max"))
@dtype_and_target_for_integer_funcs
def test_integer_binary_functions(gen_config, xp, function_name, dtype):
    sp_func, xp_func = binary_function(function_name, xp)

    smallest = np.iinfo(dtype).min
    largest = np.iinfo(dtype).max

    inp1 = xp.array([-1, 0, 1, 3, -5, -312, smallest + 1, largest], dtype=dtype)
    inp2 = xp.array([3, -5, 1, 12, 1, 11, smallest + 42, largest - 3], dtype=dtype)

    outp = xp.zeros_like(inp1)
    reference = xp_func(inp1, inp2)

    inp_field = Field.create_from_numpy_array("inp1", inp1)
    inp2_field = Field.create_from_numpy_array("inp2", inp2)
    outp_field = inp_field.new_field_with_different_name("outp")

    asms = [
        Assignment(
            outp_field.center(), sp_func(inp_field.center(), inp2_field.center())
        )
    ]
    gen_config = replace(gen_config, default_dtype=dtype)

    kernel = create_kernel(asms, gen_config)
    kfunc = kernel.compile()
    kfunc(inp1=inp1, inp2=inp2, outp=outp)

    xp.testing.assert_array_equal(outp, reference)


@pytest.mark.parametrize("a", [sp.Symbol("a"), fields("a:  float64[2d]").center])
def test_avoid_pow(a):
    x = fields("x:  float64[2d]")

    up = Assignment(x.center_vector[0], 2 * a**2 / 3)
    func = create_kernel(up)

    powers = list(
        dfs_preorder(
            func.body, lambda n: isinstance(n, PsCall) and "pow" in n.function.name
        )
    )
    assert not powers


@pytest.mark.xfail(reason="fast_div not available yet")
def test_avoid_pow_fast_div():
    x = fields("x:  float64[2d]")
    a = fields("a:  float64[2d]").center

    up = Assignment(x.center_vector[0], fast_division(1, (a**2)))
    func = create_kernel(up, config=CreateKernelConfig(target=Target.GPU))

    powers = list(
        dfs_preorder(
            func.body, lambda n: isinstance(n, PsCall) and "pow" in n.function.name
        )
    )
    assert not powers


def test_avoid_pow_move_constants():
    # At the end of the kernel creation the function move_constants_before_loop will be called
    # This function additionally contains substitutions for symbols with the same value
    # Thus it simplifies the equations again
    x = fields("x:  float64[2d]")
    a, b, c = sp.symbols("a, b, c")

    up = [
        Assignment(a, 0.0),
        Assignment(b, 0.0),
        Assignment(c, 0.0),
        Assignment(
            x.center_vector[0],
            a**2 / 18 - a * b / 6 - a / 18 + b**2 / 18 + b / 18 - c**2 / 36,
        ),
    ]
    func = create_kernel(up)

    powers = list(
        dfs_preorder(
            func.body, lambda n: isinstance(n, PsCall) and "pow" in n.function.name
        )
    )
    assert not powers
