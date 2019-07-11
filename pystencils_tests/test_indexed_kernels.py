import numpy as np

from pystencils import Assignment, Field
from pystencils.cpu import create_indexed_kernel, make_python_function


def test_indexed_kernel():
    arr = np.zeros((3, 4))
    dtype = np.dtype([('x', int), ('y', int), ('value', arr.dtype)])
    index_arr = np.zeros((3,), dtype=dtype)
    index_arr[0] = (0, 2, 3.0)
    index_arr[1] = (1, 3, 42.0)
    index_arr[2] = (2, 1, 5.0)

    indexed_field = Field.create_from_numpy_array('index', index_arr)
    normal_field = Field.create_from_numpy_array('f', arr)
    update_rule = Assignment(normal_field[0, 0], indexed_field('value'))
    ast = create_indexed_kernel([update_rule], [indexed_field])
    kernel = make_python_function(ast)
    kernel(f=arr, index=index_arr)
    for i in range(index_arr.shape[0]):
        np.testing.assert_allclose(arr[index_arr[i]['x'], index_arr[i]['y']], index_arr[i]['value'], atol=1e-13)


def test_indexed_cuda_kernel():
    try:
        import pycuda
    except ImportError:
        pycuda = None

    if pycuda:
        from pystencils.gpucuda import make_python_function
        import pycuda.gpuarray as gpuarray
        from pystencils.gpucuda.kernelcreation import created_indexed_cuda_kernel

        arr = np.zeros((3, 4))
        dtype = np.dtype([('x', int), ('y', int), ('value', arr.dtype)])
        index_arr = np.zeros((3,), dtype=dtype)
        index_arr[0] = (0, 2, 3.0)
        index_arr[1] = (1, 3, 42.0)
        index_arr[2] = (2, 1, 5.0)

        indexed_field = Field.create_from_numpy_array('index', index_arr)
        normal_field = Field.create_from_numpy_array('f', arr)
        update_rule = Assignment(normal_field[0, 0], indexed_field('value'))
        ast = created_indexed_cuda_kernel([update_rule], [indexed_field])
        kernel = make_python_function(ast)

        gpu_arr = gpuarray.to_gpu(arr)
        gpu_index_arr = gpuarray.to_gpu(index_arr)
        kernel(f=gpu_arr, index=gpu_index_arr)
        gpu_arr.get(arr)
        for i in range(index_arr.shape[0]):
            np.testing.assert_allclose(arr[index_arr[i]['x'], index_arr[i]['y']], index_arr[i]['value'], atol=1e-13)
    else:
        print("Did not run test on GPU since no pycuda is available")
