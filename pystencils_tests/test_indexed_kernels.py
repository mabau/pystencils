import sympy as sp
import numpy as np
import pytest

import pystencils as ps
from pystencils import Assignment, Field, CreateKernelConfig, create_kernel, Target
from pystencils.transformations import filtered_tree_iteration
from pystencils.typing import BasicType, FieldPointerSymbol, PointerType, TypedSymbol


@pytest.mark.parametrize('target', [ps.Target.CPU, ps.Target.GPU])
def test_indexed_kernel(target):
    if target == Target.GPU:
        pytest.importorskip("cupy")
        import cupy as cp

    arr = np.zeros((3, 4))
    dtype = np.dtype([('x', int), ('y', int), ('value', arr.dtype)])
    index_arr = np.zeros((3,), dtype=dtype)
    index_arr[0] = (0, 2, 3.0)
    index_arr[1] = (1, 3, 42.0)
    index_arr[2] = (2, 1, 5.0)

    indexed_field = Field.create_from_numpy_array('index', index_arr)
    normal_field = Field.create_from_numpy_array('f', arr)
    update_rule = Assignment(normal_field[0, 0], indexed_field('value'))

    config = CreateKernelConfig(target=target, index_fields=[indexed_field])
    ast = create_kernel([update_rule], config=config)
    kernel = ast.compile()

    if target == Target.CPU:
        kernel(f=arr, index=index_arr)
    else:
        gpu_arr = cp.asarray(arr)
        gpu_index_arr = cp.ndarray(index_arr.shape, dtype=index_arr.dtype)
        gpu_index_arr.set(index_arr)
        kernel(f=gpu_arr, index=gpu_index_arr)
        arr = gpu_arr.get()
    for i in range(index_arr.shape[0]):
        np.testing.assert_allclose(arr[index_arr[i]['x'], index_arr[i]['y']], index_arr[i]['value'], atol=1e-13)


@pytest.mark.parametrize('index_size', ("fixed", "variable"))
@pytest.mark.parametrize('array_size', ("3D", "2D", "10, 12", "13, 17, 19"))
@pytest.mark.parametrize('target', (ps.Target.CPU, ps.Target.GPU))
@pytest.mark.parametrize('dtype', ("float64", "float32"))
def test_indexed_domain_kernel(index_size, array_size, target, dtype):
    dtype = BasicType(dtype)

    f = ps.fields(f'f(1): {dtype.numpy_dtype.name}[{array_size}]')
    g = ps.fields(f'g(1): {dtype.numpy_dtype.name}[{array_size}]')

    index = TypedSymbol("index", dtype=BasicType(np.int16))
    if index_size == "variable":
        index_src = TypedSymbol("_size_src", dtype=BasicType(np.int16))
        index_dst = TypedSymbol("_size_dst", dtype=BasicType(np.int16))
    else:
        index_src = 16
        index_dst = 16
    pointer_type = PointerType(dtype, const=False, restrict=True, double_pointer=True)
    const_pointer_type = PointerType(dtype, const=True, restrict=True, double_pointer=True)

    src = sp.IndexedBase(TypedSymbol(f"_data_{f.name}", dtype=const_pointer_type), shape=index_src)
    dst = sp.IndexedBase(TypedSymbol(f"_data_{g.name}", dtype=pointer_type), shape=index_dst)

    update_rule = [ps.Assignment(FieldPointerSymbol("f", dtype, const=True), src[index]),
                   ps.Assignment(FieldPointerSymbol("g", dtype, const=False), dst[index]),
                   ps.Assignment(g.center, f.center)]

    ast = ps.create_kernel(update_rule, target=target)

    code = ps.get_code_str(ast)
    assert f"const {dtype.c_name} * RESTRICT _data_f = (({dtype.c_name} * RESTRICT const)(_data_f[index]));" in code
    assert f"{dtype.c_name} * RESTRICT  _data_g = (({dtype.c_name} * RESTRICT )(_data_g[index]));" in code

    if target == Target.CPU:
        assert code.count("for") == f.spatial_dimensions + 1

