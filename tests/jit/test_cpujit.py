import pytest

import sympy as sp
import numpy as np
from pystencils import create_kernel, Assignment, fields, Field
from pystencils.jit import CpuJit


@pytest.fixture
def cpu_jit(tmp_path) -> CpuJit:
    return CpuJit.create(objcache=tmp_path)


def test_basic_cpu_kernel(cpu_jit):
    f, g = fields("f, g: [2D]")
    asm = Assignment(f.center(), 2.0 * g.center())
    ker = create_kernel(asm)
    kfunc = cpu_jit.compile(ker)

    rng = np.random.default_rng()
    f_arr = rng.random(size=(34, 26), dtype="float64")
    g_arr = np.zeros_like(f_arr)

    kfunc(f=f_arr, g=g_arr)

    np.testing.assert_almost_equal(g_arr, 2.0 * f_arr)


def test_argument_type_error(cpu_jit):
    f, g = fields("f, g: [2D]")
    c = sp.Symbol("c")
    asm = Assignment(f.center(), c * g.center())
    ker = create_kernel(asm)
    kfunc = cpu_jit.compile(ker)

    arr_fp16 = np.zeros((23, 12), dtype="float16")
    arr_fp32 = np.zeros((23, 12), dtype="float32")
    arr_fp64 = np.zeros((23, 12), dtype="float64")

    with pytest.raises(TypeError):
        kfunc(f=arr_fp32, g=arr_fp64, c=2.0)

    with pytest.raises(TypeError):
        kfunc(f=arr_fp64, g=arr_fp32, c=2.0)

    with pytest.raises(TypeError):
        kfunc(f=arr_fp16, g=arr_fp16, c=2.0)

    #   Wrong scalar types are OK, though
    kfunc(f=arr_fp64, g=arr_fp64, c=np.float16(1.0))


def test_fixed_shape(cpu_jit):
    a = np.zeros((12, 23), dtype="float64")
    b = np.zeros((13, 21), dtype="float64")
    
    f = Field.create_from_numpy_array("f", a)
    g = Field.create_from_numpy_array("g", a)

    asm = Assignment(f.center(), 2.0 * g.center())
    ker = create_kernel(asm)
    kfunc = cpu_jit.compile(ker)

    kfunc(f=a, g=a)

    with pytest.raises(ValueError):
        kfunc(f=b, g=a)

    with pytest.raises(ValueError):
        kfunc(f=a, g=b)


def test_fixed_index_shape(cpu_jit):
    f, g = fields("f(3), g(2, 2): [2D]")

    asm = Assignment(f.center(1), g.center(0, 0) + g.center(0, 1) + g.center(1, 0) + g.center(1, 1))
    ker = create_kernel(asm)
    kfunc = cpu_jit.compile(ker)

    f_arr = np.zeros((12, 14, 3))
    g_arr = np.zeros((12, 14, 2, 2))
    kfunc(f=f_arr, g=g_arr)

    with pytest.raises(ValueError):
        f_arr = np.zeros((12, 14, 2))
        g_arr = np.zeros((12, 14, 2, 2))
        kfunc(f=f_arr, g=g_arr)

    with pytest.raises(ValueError):
        f_arr = np.zeros((12, 14, 3))
        g_arr = np.zeros((12, 14, 4))
        kfunc(f=f_arr, g=g_arr)

    with pytest.raises(ValueError):
        f_arr = np.zeros((12, 14, 3))
        g_arr = np.zeros((12, 14, 1, 3))
        kfunc(f=f_arr, g=g_arr)
