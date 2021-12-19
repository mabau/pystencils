import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import sympy as sp
from scipy.ndimage import convolve

from pystencils import Assignment, Field, fields, CreateKernelConfig, create_kernel, Target
from pystencils.gpucuda import BlockIndexing
from pystencils.simp import sympy_cse_on_assignment_list
from pystencils.slicing import add_ghost_layers, make_slice, remove_ghost_layers


def test_averaging_kernel():
    size = (40, 55)
    src_arr = np.random.rand(*size)
    src_arr = add_ghost_layers(src_arr)
    dst_arr = np.zeros_like(src_arr)
    src_field = Field.create_from_numpy_array('src', src_arr)
    dst_field = Field.create_from_numpy_array('dst', dst_arr)

    update_rule = Assignment(dst_field[0, 0],
                             (src_field[0, 1] + src_field[0, -1] + src_field[1, 0] + src_field[-1, 0]) / 4)

    config = CreateKernelConfig(target=Target.GPU)
    ast = create_kernel(sympy_cse_on_assignment_list([update_rule]), config=config)
    kernel = ast.compile()

    gpu_src_arr = gpuarray.to_gpu(src_arr)
    gpu_dst_arr = gpuarray.to_gpu(dst_arr)
    kernel(src=gpu_src_arr, dst=gpu_dst_arr)
    gpu_dst_arr.get(dst_arr)

    stencil = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4.0
    reference = convolve(remove_ghost_layers(src_arr), stencil, mode='constant', cval=0.0)
    reference = add_ghost_layers(reference)
    np.testing.assert_almost_equal(reference, dst_arr)


def test_variable_sized_fields():
    src_field = Field.create_generic('src', spatial_dimensions=2)
    dst_field = Field.create_generic('dst', spatial_dimensions=2)

    update_rule = Assignment(dst_field[0, 0],
                             (src_field[0, 1] + src_field[0, -1] + src_field[1, 0] + src_field[-1, 0]) / 4)

    config = CreateKernelConfig(target=Target.GPU)
    ast = create_kernel(sympy_cse_on_assignment_list([update_rule]), config=config)
    kernel = ast.compile()

    size = (3, 3)
    src_arr = np.random.rand(*size)
    src_arr = add_ghost_layers(src_arr)
    dst_arr = np.zeros_like(src_arr)

    gpu_src_arr = gpuarray.to_gpu(src_arr)
    gpu_dst_arr = gpuarray.to_gpu(dst_arr)
    kernel(src=gpu_src_arr, dst=gpu_dst_arr)
    gpu_dst_arr.get(dst_arr)

    stencil = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4.0
    reference = convolve(remove_ghost_layers(src_arr), stencil, mode='constant', cval=0.0)
    reference = add_ghost_layers(reference)
    np.testing.assert_almost_equal(reference, dst_arr)


def test_multiple_index_dimensions():
    """Sums along the last axis of a numpy array"""
    src_size = (7, 6, 4)
    dst_size = src_size[:2]
    src_arr = np.asfortranarray(np.random.rand(*src_size))
    dst_arr = np.zeros(dst_size)

    src_field = Field.create_from_numpy_array('src', src_arr, index_dimensions=1)
    dst_field = Field.create_from_numpy_array('dst', dst_arr, index_dimensions=0)

    offset = (-2, -1)
    update_rule = Assignment(dst_field[0, 0],
                             sum([src_field[offset[0], offset[1]](i) for i in range(src_size[-1])]))

    config = CreateKernelConfig(target=Target.GPU)
    ast = create_kernel([update_rule], config=config)
    kernel = ast.compile()

    gpu_src_arr = gpuarray.to_gpu(src_arr)
    gpu_dst_arr = gpuarray.to_gpu(dst_arr)
    kernel(src=gpu_src_arr, dst=gpu_dst_arr)
    gpu_dst_arr.get(dst_arr)

    reference = np.zeros_like(dst_arr)
    gl = np.max(np.abs(np.array(offset, dtype=int)))
    for x in range(gl, src_size[0]-gl):
        for y in range(gl, src_size[1]-gl):
            reference[x, y] = sum([src_arr[x+offset[0], y+offset[1], i] for i in range(src_size[2])])

    np.testing.assert_almost_equal(reference, dst_arr)


def test_ghost_layer():
    size = (6, 5)
    src_arr = np.ones(size)
    dst_arr = np.zeros_like(src_arr)
    src_field = Field.create_from_numpy_array('src', src_arr, index_dimensions=0)
    dst_field = Field.create_from_numpy_array('dst', dst_arr, index_dimensions=0)

    update_rule = Assignment(dst_field[0, 0], src_field[0, 0])
    ghost_layers = [(1, 2), (2, 1)]

    config = CreateKernelConfig(target=Target.GPU, ghost_layers=ghost_layers, gpu_indexing="line")
    ast = create_kernel(sympy_cse_on_assignment_list([update_rule]), config=config)
    kernel = ast.compile()

    gpu_src_arr = gpuarray.to_gpu(src_arr)
    gpu_dst_arr = gpuarray.to_gpu(dst_arr)
    kernel(src=gpu_src_arr, dst=gpu_dst_arr)
    gpu_dst_arr.get(dst_arr)

    reference = np.zeros_like(src_arr)
    reference[ghost_layers[0][0]:-ghost_layers[0][1], ghost_layers[1][0]:-ghost_layers[1][1]] = 1
    np.testing.assert_equal(reference, dst_arr)


def test_setting_value():
    arr_cpu = np.arange(25, dtype=np.float64).reshape(5, 5)
    arr_gpu = gpuarray.to_gpu(arr_cpu)

    iteration_slice = make_slice[:, :]
    f = Field.create_generic("f", 2)
    update_rule = [Assignment(f(0), sp.Symbol("value"))]

    config = CreateKernelConfig(target=Target.GPU, gpu_indexing="line", iteration_slice=iteration_slice)
    ast = create_kernel(sympy_cse_on_assignment_list(update_rule), config=config)
    kernel = ast.compile()

    kernel(f=arr_gpu, value=np.float64(42.0))
    np.testing.assert_equal(arr_gpu.get(), np.ones((5, 5)) * 42.0)


def test_periodicity():
    from pystencils.gpucuda.periodicity import get_periodic_boundary_functor as periodic_gpu
    from pystencils.slicing import get_periodic_boundary_functor as periodic_cpu

    arr_cpu = np.arange(50, dtype=np.float64).reshape(5, 5, 2)
    arr_gpu = gpuarray.to_gpu(arr_cpu)

    periodicity_stencil = [(1, 0), (-1, 0), (1, 1)]
    periodic_gpu_kernel = periodic_gpu(periodicity_stencil, (5, 5), 1, 2)
    periodic_cpu_kernel = periodic_cpu(periodicity_stencil)

    cpu_result = np.copy(arr_cpu)
    periodic_cpu_kernel(cpu_result)

    gpu_result = np.copy(arr_cpu)
    periodic_gpu_kernel(pdfs=arr_gpu)
    arr_gpu.get(gpu_result)
    np.testing.assert_equal(cpu_result, gpu_result)


def test_block_indexing():
    f = fields("f: [3D]")
    bi = BlockIndexing(f, make_slice[:, :, :], block_size=(16, 8, 2), permute_block_size_dependent_on_layout=False)
    assert bi.call_parameters((3, 2, 32))['block'] == (3, 2, 32)
    assert bi.call_parameters((32, 2, 32))['block'] == (16, 2, 8)

    bi = BlockIndexing(f, make_slice[:, :, :], block_size=(32, 1, 1), permute_block_size_dependent_on_layout=False)
    assert bi.call_parameters((1, 16, 16))['block'] == (1, 16, 2)

    bi = BlockIndexing(f, make_slice[:, :, :], block_size=(16, 8, 2), maximum_block_size="auto")
    # This function should be used if number of needed registers is known. Can be determined with func.num_regs
    blocks = bi.limit_block_size_by_register_restriction([1024, 1024, 1], 1000)

    assert sum(blocks) < sum([1024, 1024, 1])
