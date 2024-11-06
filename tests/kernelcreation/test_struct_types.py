import numpy as np
import pytest

from pystencils import Assignment, Field, create_kernel


@pytest.mark.parametrize("order", ['c', 'f'])
@pytest.mark.parametrize("align", [True, False])
def test_fixed_sized_field(order, align):
    if not align:
        pytest.xfail("Non-Aligned structs not supported")
    dt = np.dtype([('e1', np.float32), ('e2', np.double), ('e3', np.double)], align=align)
    arr = np.zeros((3, 2), dtype=dt, order=order)

    f = Field.create_from_numpy_array("f", arr)
    d = Field.create_from_numpy_array("d", arr)
    update_rules = [Assignment(d[0, 0]['e2'], f[0, 0]['e3'])]
    result = arr.copy(order=order)
    assert result.strides == arr.strides
    arr['e2'] = 0
    arr['e3'] = np.random.rand(3, 2)

    kernel = create_kernel(update_rules).compile()
    kernel(f=arr, d=result)
    np.testing.assert_almost_equal(result['e2'], arr['e3'])
    np.testing.assert_equal(arr['e2'], np.zeros((3, 2)))


@pytest.mark.parametrize("order", ['c', 'f'])
@pytest.mark.parametrize("align", [True, False])
def test_variable_sized_field(order, align):
    if not align:
        pytest.xfail("Non-Aligned structs not supported")
    dt = np.dtype([('e1', np.float32), ('e2', np.double), ('e3', np.double)], align=align)

    f = Field.create_generic("f", 2, dt, layout=order)
    d = Field.create_generic("d", 2, dt, layout=order)
    update_rules = [Assignment(d[0, 0]['e2'], f[0, 0]['e3'])]

    arr = np.zeros((3, 2), dtype=dt, order=order)
    result = arr.copy(order=order)

    arr['e2'] = 0
    arr['e3'] = np.random.rand(3, 2)

    kernel = create_kernel(update_rules).compile()
    kernel(f=arr, d=result)
    np.testing.assert_almost_equal(result['e2'], arr['e3'])
    np.testing.assert_equal(arr['e2'], np.zeros((3, 2)))
