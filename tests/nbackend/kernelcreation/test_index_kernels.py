import numpy as np

from pystencils import Assignment, Field, FieldType, AssignmentCollection
from pystencils.kernelcreation import create_kernel, CreateKernelConfig


def test_indexed_kernel():
    arr = np.zeros((3, 4))
    dtype = np.dtype([('x', int), ('y', int), ('value', arr.dtype)])
    index_arr = np.zeros((3,), dtype=dtype)
    index_arr[0] = (0, 2, 3.0)
    index_arr[1] = (1, 3, 42.0)
    index_arr[2] = (2, 1, 5.0)

    index_field = Field.create_from_numpy_array('index', index_arr, field_type=FieldType.INDEXED)
    normal_field = Field.create_from_numpy_array('f', arr)
    update_rule = AssignmentCollection([
        Assignment(normal_field[0, 0], index_field('value'))
    ])

    options = CreateKernelConfig(index_field=index_field)
    ast = create_kernel(update_rule, options)
    kernel = ast.compile()

    kernel(f=arr, index=index_arr)

    for i in range(index_arr.shape[0]):
        np.testing.assert_allclose(arr[index_arr[i]['x'], index_arr[i]['y']], index_arr[i]['value'], atol=1e-13)
