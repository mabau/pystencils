import sympy as sp
import numpy as np
import pytest

from pystencils import create_kernel, CreateKernelConfig, Target, Assignment, Field

UNARY_FUNCTIONS = {
    "exp": (sp.exp, np.exp),
    "sin": (sp.sin, np.sin),
    "cos": (sp.cos, np.cos),
    "tan": (sp.tan, np.tan),
    "abs": (sp.Abs, np.abs),
}

BINARY_FUNCTIONS = {
    "min": (sp.Min, np.fmin),
    "max": (sp.Max, np.fmax),
    "pow": (sp.Pow, np.power),
}


@pytest.mark.parametrize("target", (Target.GenericCPU,))
@pytest.mark.parametrize("function_name", UNARY_FUNCTIONS.keys())
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_unary_functions(target, function_name, dtype):
    sp_func, np_func = UNARY_FUNCTIONS[function_name]
    resolution: dtype = np.finfo(dtype).resolution

    inp = np.array(
        [[0.1, 0.2, 0.3], [-0.8, -1.6, -12.592], [np.pi, np.e, 0.0]], dtype=dtype
    )
    outp = np.zeros_like(inp)

    reference = np_func(inp)

    inp_field = Field.create_from_numpy_array("inp", inp)
    outp_field = inp_field.new_field_with_different_name("outp")

    asms = [Assignment(outp_field.center(), sp_func(inp_field.center()))]
    gen_config = CreateKernelConfig(target=target, default_dtype=dtype)

    kernel = create_kernel(asms, gen_config)
    kfunc = kernel.compile()
    kfunc(inp=inp, outp=outp)

    np.testing.assert_allclose(outp, reference, rtol=resolution)


@pytest.mark.parametrize("target", (Target.GenericCPU,))
@pytest.mark.parametrize("function_name", BINARY_FUNCTIONS.keys())
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_binary_functions(target, function_name, dtype):
    sp_func, np_func = BINARY_FUNCTIONS[function_name]
    resolution: dtype = np.finfo(dtype).resolution

    inp = np.array(
        [[0.1, 0.2, 0.3], [-0.8, -1.6, -12.592], [np.pi, np.e, 0.0]], dtype=dtype
    )
    inp2 = np.array(
        [[3.1, -0.5, 21.409], [11.0, 1.0, -14e3], [2.0 * np.pi, - np.e, 0.0]], dtype=dtype
    )
    outp = np.zeros_like(inp)

    reference = np_func(inp, inp2)

    inp_field = Field.create_from_numpy_array("inp", inp)
    inp2_field = Field.create_from_numpy_array("inp2", inp)
    outp_field = inp_field.new_field_with_different_name("outp")

    asms = [Assignment(outp_field.center(), sp_func(inp_field.center(), inp2_field.center()))]
    gen_config = CreateKernelConfig(target=target, default_dtype=dtype)

    kernel = create_kernel(asms, gen_config)
    kfunc = kernel.compile()
    kfunc(inp=inp, inp2=inp2, outp=outp)

    np.testing.assert_allclose(outp, reference, rtol=resolution)
