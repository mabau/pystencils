import numpy as np
import sympy as sp
import pytest

import pystencils as ps
from pystencils.astnodes import Block, Conditional, SympyAssignment
from pystencils.backends.simd_instruction_sets import get_supported_instruction_sets, get_vector_instruction_set
from pystencils.enums import Target
from pystencils.cpu.vectorization import vec_all, vec_any
from pystencils.node_collection import NodeCollection

supported_instruction_sets = get_supported_instruction_sets() if get_supported_instruction_sets() else []


@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
@pytest.mark.parametrize('dtype', ('float32', 'float64'))
def test_vec_any(instruction_set, dtype):
    if instruction_set in ['sve', 'rvv']:
        width = 4  # we don't know the actual value
    else:
        width = get_vector_instruction_set(dtype, instruction_set)['width']
    data_arr = np.zeros((4 * width, 4 * width), dtype=dtype)

    data_arr[3:9, 1:3 * width - 1] = 1.0
    data = ps.fields(f"data: {dtype}[2D]", data=data_arr)

    c = [
        SympyAssignment(sp.Symbol("t1"), vec_any(data.center() > 0.0)),
        Conditional(vec_any(data.center() > 0.0), Block([SympyAssignment(data.center(), 2.0)]))
    ]

    assignmets = NodeCollection(c)
    ast = ps.create_kernel(assignments=assignmets, target=ps.Target.CPU,
                           cpu_vectorize_info={'instruction_set': instruction_set})
    kernel = ast.compile()
    kernel(data=data_arr)
    if instruction_set in ['sve', 'rvv']:
        # we only know that the first value has changed
        np.testing.assert_equal(data_arr[3:9, :3 * width - 1], 2.0)
    else:
        np.testing.assert_equal(data_arr[3:9, :3 * width], 2.0)


@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
@pytest.mark.parametrize('dtype', ('float32', 'float64'))
def test_vec_all(instruction_set, dtype):
    if instruction_set in ['sve', 'rvv']:
        width = 1000  # we don't know the actual value, need something guaranteed larger than vector
    else:
        width = get_vector_instruction_set(dtype, instruction_set)['width']
    data_arr = np.zeros((4 * width, 4 * width), dtype=dtype)

    data_arr[3:9, 1:3 * width - 1] = 1.0
    data = ps.fields(f"data: {dtype}[2D]", data=data_arr)

    c = [Conditional(vec_all(data.center() > 0.0), Block([SympyAssignment(data.center(), 2.0)]))]
    assignmets = NodeCollection(c)
    ast = ps.create_kernel(assignmets, target=Target.CPU,
                           cpu_vectorize_info={'instruction_set': instruction_set})
    kernel = ast.compile()
    kernel(data=data_arr)
    if instruction_set in ['sve', 'rvv']:
        # we only know that some values in the middle have been replaced
        assert np.all(data_arr[3:9, :2] <= 1.0)
        assert np.any(data_arr[3:9, 2:] == 2.0)
    else:
        np.testing.assert_equal(data_arr[3:9, :1], 0.0)
        np.testing.assert_equal(data_arr[3:9, 1:width], 1.0)
        np.testing.assert_equal(data_arr[3:9, width:2 * width], 2.0)
        np.testing.assert_equal(data_arr[3:9, 2 * width:3 * width - 1], 1.0)
        np.testing.assert_equal(data_arr[3:9, 3 * width - 1:], 0.0)


@pytest.mark.skipif(not supported_instruction_sets, reason='cannot detect CPU instruction set')
def test_boolean_before_loop():
    t1, t2 = sp.symbols('t1, t2')
    f_arr = np.ones((10, 10))
    g_arr = np.zeros_like(f_arr)
    f, g = ps.fields("f, g : double[2D]", f=f_arr, g=g_arr)

    a = [
        ps.Assignment(t1, t2 > 0),
        ps.Assignment(g[0, 0],
                      sp.Piecewise((f[0, 0], t1), (42, True)))
    ]
    ast = ps.create_kernel(a, cpu_vectorize_info={'instruction_set': supported_instruction_sets[-1]})
    kernel = ast.compile()
    kernel(f=f_arr, g=g_arr, t2=1.0)
    # print(g)
    np.testing.assert_array_equal(g_arr, 1.0)
    kernel(f=f_arr, g=g_arr, t2=-1.0)
    np.testing.assert_array_equal(g_arr, 42.0)


@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
@pytest.mark.parametrize('dtype', ('float32', 'float64'))
def test_vec_maskstore(instruction_set, dtype):
    data_arr = np.zeros((16, 16), dtype=dtype)
    data_arr[3:-3, 3:-3] = 1.0
    data = ps.fields(f"data: {dtype}[2D]", data=data_arr)

    c = [Conditional(data.center() < 1.0, Block([SympyAssignment(data.center(), 2.0)]))]

    assignmets = NodeCollection(c)
    config = ps.CreateKernelConfig(cpu_vectorize_info={'instruction_set': instruction_set}, default_number_float=dtype)
    ast = ps.create_kernel(assignmets, config=config)
    print(ps.get_code_str(ast))
    kernel = ast.compile()
    kernel(data=data_arr)
    np.testing.assert_equal(data_arr[:3, :], 2.0)
    np.testing.assert_equal(data_arr[-3:, :], 2.0)
    np.testing.assert_equal(data_arr[:, :3], 2.0)
    np.testing.assert_equal(data_arr[:, -3:], 2.0)
    np.testing.assert_equal(data_arr[3:-3, 3:-3], 1.0)
