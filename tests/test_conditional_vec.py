import numpy as np
import sympy as sp
import pytest

import pystencils as ps
from pystencils.alignedarray import aligned_zeros
from pystencils.astnodes import Block, Conditional, SympyAssignment
from pystencils.backends.simd_instruction_sets import get_supported_instruction_sets, get_vector_instruction_set
from pystencils.enums import Target
from pystencils.cpu.vectorization import vec_all, vec_any
from pystencils.node_collection import NodeCollection

supported_instruction_sets = get_supported_instruction_sets() if get_supported_instruction_sets() else []


@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
@pytest.mark.parametrize('dtype', ('float32', 'float64'))
def test_vec_any(instruction_set, dtype):
    if instruction_set in ['sve', 'sve2', 'sme', 'rvv']:
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
    if instruction_set in ['sve', 'sve2', 'sme', 'rvv']:
        # we only know that the first value has changed
        np.testing.assert_equal(data_arr[3:9, :3 * width - 1], 2.0)
    else:
        np.testing.assert_equal(data_arr[3:9, :3 * width], 2.0)


@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
@pytest.mark.parametrize('dtype', ('float32', 'float64'))
def test_vec_all(instruction_set, dtype):
    if instruction_set in ['sve', 'sve2', 'sme', 'rvv']:
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
    if instruction_set in ['sve', 'sve2', 'sme', 'rvv']:
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
@pytest.mark.parametrize('nontemporal', [False, True])
@pytest.mark.parametrize('aligned', [False, True])
def test_vec_maskstore(instruction_set, dtype, nontemporal, aligned):
    data_arr = (aligned_zeros if aligned else np.zeros)((16, 16), dtype=dtype)
    data_arr[3:-3, 3:-3] = 1.0
    data = ps.fields(f"data: {dtype}[2D]", data=data_arr)

    c = [Conditional(data.center() < 1.0, Block([SympyAssignment(data.center(), 2.0)]))]

    assignmets = NodeCollection(c)
    config = ps.CreateKernelConfig(cpu_vectorize_info={'instruction_set': instruction_set,
                                                       'nontemporal': nontemporal,
                                                       'assume_aligned': aligned},
                                   default_number_float=dtype)
    ast = ps.create_kernel(assignmets, config=config)
    if 'maskStore' in ast.instruction_set:
        instruction = 'maskStream' if nontemporal and 'maskStream' in ast.instruction_set else (
                      'maskStoreA' if aligned and 'maskStoreA' in ast.instruction_set else 'maskStore')
        assert ast.instruction_set[instruction].split('{')[0] in ps.get_code_str(ast)
    print(ps.get_code_str(ast))
    kernel = ast.compile()
    kernel(data=data_arr)
    np.testing.assert_equal(data_arr[:3, :], 2.0)
    np.testing.assert_equal(data_arr[-3:, :], 2.0)
    np.testing.assert_equal(data_arr[:, :3], 2.0)
    np.testing.assert_equal(data_arr[:, -3:], 2.0)
    np.testing.assert_equal(data_arr[3:-3, 3:-3], 1.0)


@pytest.mark.parametrize('instruction_set', supported_instruction_sets)
@pytest.mark.parametrize('dtype', ('float32', 'float64'))
@pytest.mark.parametrize('nontemporal', [False, True])
def test_vec_maskscatter(instruction_set, dtype, nontemporal):
    data_arr = np.zeros((16, 16), dtype=dtype)
    data_arr[3:-3, 3:-3] = 1.0
    data = ps.fields(f"data: {dtype}[2D]")

    c = [Conditional(data.center() < 1.0, Block([SympyAssignment(data.center(), 2.0)]))]

    assignmets = NodeCollection(c)
    config = ps.CreateKernelConfig(cpu_vectorize_info={'instruction_set': instruction_set,
                                                       'nontemporal': nontemporal},
                                   default_number_float=dtype)
    if 'maskStoreS' not in get_vector_instruction_set(dtype, instruction_set) \
            and not instruction_set.startswith('sve'):
        with pytest.warns(UserWarning) as warn:
            ast = ps.create_kernel(assignmets, config=config)
            assert 'Could not vectorize loop' in warn[0].message.args[0]
    else:
        with pytest.warns(None) as warn:
            ast = ps.create_kernel(assignmets, config=config)
            assert len(warn) == 0
        instruction = 'maskStreamS' if nontemporal and 'maskStreamS' in ast.instruction_set else 'maskStoreS'
        assert ast.instruction_set[instruction].split('{')[0] in ps.get_code_str(ast)
    print(ps.get_code_str(ast))
    kernel = ast.compile()
    kernel(data=data_arr)
    np.testing.assert_equal(data_arr[:3, :], 2.0)
    np.testing.assert_equal(data_arr[-3:, :], 2.0)
    np.testing.assert_equal(data_arr[:, :3], 2.0)
    np.testing.assert_equal(data_arr[:, -3:], 2.0)
    np.testing.assert_equal(data_arr[3:-3, 3:-3], 1.0)
