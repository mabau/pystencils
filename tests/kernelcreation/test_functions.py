import sympy as sp
import numpy as np
import pytest

from pystencils import fields, create_kernel, CreateKernelConfig, Target, Assignment, Field
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


@pytest.mark.parametrize("target", (Target.GenericCPU, Target.CUDA))
@pytest.mark.parametrize(
    "function_name",
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
        "abs",
        "floor",
        "ceil",
    ),
)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_unary_functions(target, function_name, dtype):
    if target == Target.CUDA:
        xp = pytest.importorskip("cupy")
    else:
        xp = np

    sp_func, xp_func = unary_function(function_name, xp)
    resolution: dtype = np.finfo(dtype).resolution

    inp = xp.array(
        [[0.1, 0.2, 0.3], [-0.8, -1.6, -12.592], [xp.pi, xp.e, 0.0]], dtype=dtype
    )
    outp = xp.zeros_like(inp)

    reference = xp_func(inp)

    inp_field = Field.create_from_numpy_array("inp", inp)
    outp_field = inp_field.new_field_with_different_name("outp")

    asms = [Assignment(outp_field.center(), sp_func(inp_field.center()))]
    gen_config = CreateKernelConfig(target=target, default_dtype=dtype)

    kernel = create_kernel(asms, gen_config)
    kfunc = kernel.compile()
    kfunc(inp=inp, outp=outp)

    xp.testing.assert_allclose(outp, reference, rtol=resolution)


@pytest.mark.parametrize("target", (Target.GenericCPU, Target.CUDA))
@pytest.mark.parametrize("function_name", ("min", "max", "pow", "atan2"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_binary_functions(target, function_name, dtype):
    if target == Target.CUDA:
        xp = pytest.importorskip("cupy")
    else:
        xp = np

    sp_func, np_func = binary_function(function_name, xp)
    resolution: dtype = np.finfo(dtype).resolution

    inp = xp.array(
        [[0.1, 0.2, 0.3], [-0.8, -1.6, -12.592], [xp.pi, xp.e, 0.0]], dtype=dtype
    )
    inp2 = xp.array(
        [[3.1, -0.5, 21.409], [11.0, 1.0, -14e3], [2.0 * xp.pi, -xp.e, 0.0]],
        dtype=dtype,
    )
    outp = xp.zeros_like(inp)

    reference = np_func(inp, inp2)

    inp_field = Field.create_from_numpy_array("inp", inp)
    inp2_field = Field.create_from_numpy_array("inp2", inp)
    outp_field = inp_field.new_field_with_different_name("outp")

    asms = [
        Assignment(
            outp_field.center(), sp_func(inp_field.center(), inp2_field.center())
        )
    ]
    gen_config = CreateKernelConfig(target=target, default_dtype=dtype)

    kernel = create_kernel(asms, gen_config)
    kfunc = kernel.compile()
    kfunc(inp=inp, inp2=inp2, outp=outp)

    xp.testing.assert_allclose(outp, reference, rtol=resolution)


@pytest.mark.parametrize('a', [sp.Symbol('a'), fields('a:  float64[2d]').center])
def test_avoid_pow(a):
    x = fields('x:  float64[2d]')

    up = Assignment(x.center_vector[0], 2 * a ** 2 / 3)
    func = create_kernel(up)

    powers = list(dfs_preorder(func.body, lambda n: isinstance(n, PsCall) and "pow" in n.function.name))
    assert not powers


@pytest.mark.xfail(reason="fast_div not available yet")
def test_avoid_pow_fast_div():
    x = fields('x:  float64[2d]')
    a = fields('a:  float64[2d]').center

    up = Assignment(x.center_vector[0], fast_division(1, (a**2)))
    func = create_kernel(up, config=CreateKernelConfig(target=Target.GPU))

    powers = list(dfs_preorder(func.body, lambda n: isinstance(n, PsCall) and "pow" in n.function.name))
    assert not powers


def test_avoid_pow_move_constants():
    # At the end of the kernel creation the function move_constants_before_loop will be called
    # This function additionally contains substitutions for symbols with the same value
    # Thus it simplifies the equations again
    x = fields('x:  float64[2d]')
    a, b, c = sp.symbols("a, b, c")

    up = [Assignment(a, 0.0),
          Assignment(b, 0.0),
          Assignment(c, 0.0),
          Assignment(x.center_vector[0], a**2/18 - a*b/6 - a/18 + b**2/18 + b/18 - c**2/36)]
    func = create_kernel(up)

    powers = list(dfs_preorder(func.body, lambda n: isinstance(n, PsCall) and "pow" in n.function.name))
    assert not powers
