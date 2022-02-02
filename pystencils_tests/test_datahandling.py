import os
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

import pystencils as ps
from pystencils import create_data_handling, create_kernel
from pystencils.datahandling.pycuda import PyCudaArrayHandler
from pystencils.enums import Target

try:
    import pytest
except ImportError:
    import unittest.mock
    pytest = unittest.mock.MagicMock()

SCRIPT_FOLDER = Path(__file__).parent.absolute()
INPUT_FOLDER = SCRIPT_FOLDER / "test_data"


def basic_iteration(dh):
    dh.add_array('basic_iter_test_gl_default')
    dh.add_array('basic_iter_test_gl_3', ghost_layers=3)

    for b in dh.iterate():
        assert b.shape == b['basic_iter_test_gl_3'].shape
        assert b.shape == b['basic_iter_test_gl_default'].shape


def access_and_gather(dh, domain_size):
    dh.add_array('f1', dtype=np.dtype(np.int8))
    dh.add_array_like('f2', 'f1')
    dh.add_array('v1', values_per_cell=3, dtype=np.int64, ghost_layers=2)
    dh.add_array_like('v2', 'v1')

    dh.swap('f1', 'f2')
    dh.swap('v1', 'v2')

    # Check symbolic field properties
    assert dh.fields.f1.index_dimensions == 0
    assert dh.fields.f1.spatial_dimensions == len(domain_size)
    assert dh.fields.f1.dtype.numpy_dtype == np.int8

    assert dh.fields.v1.index_dimensions == 1
    assert dh.fields.v1.spatial_dimensions == len(domain_size)
    assert dh.fields.v1.dtype.numpy_dtype == np.int64

    for b in dh.iterate(ghost_layers=0):
        val = sum(b.cell_index_arrays)
        np.copyto(b['f1'], val)
        for i, coord_arr in enumerate(b.cell_index_arrays):
            np.copyto(b['v1'][..., i], coord_arr)

    full_arr = dh.gather_array('v1')
    if full_arr is not None:
        expected_shape = domain_size + (3,)
        assert full_arr.shape == expected_shape
        for x in range(full_arr.shape[0]):
            for y in range(full_arr.shape[1]):
                if len(domain_size) == 3:
                    for z in range(full_arr.shape[2]):
                        assert full_arr[x, y, z, 0] == x
                        assert full_arr[x, y, z, 1] == y
                        assert full_arr[x, y, z, 2] == z
                else:
                    assert len(domain_size) == 2
                    assert full_arr[x, y, 0] == x
                    assert full_arr[x, y, 1] == y

    full_arr = dh.gather_array('f1')
    if full_arr is not None:
        expected_shape = domain_size
        assert full_arr.shape == expected_shape
        for x in range(full_arr.shape[0]):
            for y in range(full_arr.shape[1]):
                if len(domain_size) == 3:
                    for z in range(full_arr.shape[2]):
                        assert full_arr[x, y, z] == x + y + z
                else:
                    assert len(domain_size) == 2
                    assert full_arr[x, y] == x + y


def synchronization(dh, test_gpu=False):
    field_name = 'comm_field_test'
    if test_gpu:
        try:
            from pycuda import driver
            import pycuda.autoinit
        except ImportError:
            return
        field_name += 'Gpu'

    dh.add_array(field_name, ghost_layers=1, dtype=np.int8, cpu=True, gpu=test_gpu)

    # initialize everything with 1
    for b in dh.iterate(ghost_layers=1):
        b[field_name].fill(1)
    for b in dh.iterate(ghost_layers=0):
        b[field_name].fill(42)

    if test_gpu:
        dh.to_gpu(field_name)
        dh.synchronization_function_gpu(field_name)()
    else:
        dh.synchronization_function_cpu(field_name)()


    if test_gpu:
        dh.to_cpu(field_name)

    for b in dh.iterate(ghost_layers=1):
        np.testing.assert_equal(42, b[field_name])


def kernel_execution_jacobi(dh, target):

    test_gpu = target == Target.GPU
    dh.add_array('f', gpu=test_gpu)
    dh.add_array('tmp', gpu=test_gpu)

    if test_gpu:
        assert dh.is_on_gpu('f')
        assert dh.is_on_gpu('tmp')

    stencil_2d = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    stencil_3d = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    stencil = stencil_2d if dh.dim == 2 else stencil_3d

    @ps.kernel
    def jacobi():
        dh.fields.tmp.center @= sum(dh.fields.f.neighbors(stencil)) / len(stencil)

    kernel = create_kernel(jacobi, config=ps.CreateKernelConfig(target=target)).compile()
    for b in dh.iterate(ghost_layers=1):
        b['f'].fill(42)
    dh.run_kernel(kernel)
    for b in dh.iterate(ghost_layers=0):
        np.testing.assert_equal(b['f'], 42)


def vtk_output(dh):
    pytest.importorskip('pyevtk')
    dh.add_array('scalar_field')
    dh.add_array('vector_field', values_per_cell=dh.dim)
    dh.add_array('multiple_scalar_field', values_per_cell=9)
    dh.add_array('flag_field', dtype=np.uint16)

    fields_names = ['scalar_field', 'vector_field', 'multiple_scalar_field', 'flag_field']
    with TemporaryDirectory() as tmp_dir:
        writer1 = dh.create_vtk_writer(os.path.join(tmp_dir, "out1"), fields_names, ghost_layers=True)
        writer2 = dh.create_vtk_writer(os.path.join(tmp_dir, "out2"), fields_names, ghost_layers=False)
        masks_to_name = {1: 'flag1', 5: 'some_mask'}
        writer3 = dh.create_vtk_writer_for_flag_array(os.path.join(tmp_dir, "out3"), 'flag_field', masks_to_name)
        writer1(1)
        writer2(1)
        writer3(1)


def reduction(dh):
    float_seq = [1.0, 2.0, 3.0]
    int_seq = [1, 2, 3]
    for op in ('min', 'max', 'sum'):
        assert (dh.reduce_float_sequence(float_seq, op) == float_seq).all()
        assert (dh.reduce_int_sequence(int_seq, op) == int_seq).all()


def test_symbolic_fields():
    dh = create_data_handling(domain_size=(5, 7))
    dh.add_array('f1', values_per_cell=dh.dim)
    assert dh.fields['f1'].spatial_dimensions == dh.dim
    assert dh.fields['f1'].index_dimensions == 1

    dh.add_array_like("f_tmp", "f1", latex_name=r"f_{tmp}")
    assert dh.fields['f_tmp'].spatial_dimensions == dh.dim
    assert dh.fields['f_tmp'].index_dimensions == 1

    dh.swap('f1', 'f_tmp')


def test_access():
    for domain_shape in [(2, 3, 4), (2, 4)]:
        for f_size in (1, 4):
            dh = create_data_handling(domain_size=domain_shape)
            dh.add_array('f1', values_per_cell=f_size)
            assert dh.dim == len(domain_shape)

            for b in dh.iterate(ghost_layers=1):
                if f_size > 1:
                    assert b['f1'].shape == tuple(ds+2 for ds in domain_shape) + (f_size,)
                else:
                    assert b['f1'].shape == tuple(ds + 2 for ds in domain_shape)

            for b in dh.iterate(ghost_layers=0):
                if f_size > 1:
                    assert b['f1'].shape == domain_shape + (f_size,)
                else:
                    assert b['f1'].shape == domain_shape


def test_access_and_gather():
    for domain_shape in [(2, 2, 3), (2, 3)]:
        dh = create_data_handling(domain_size=domain_shape, periodicity=True)
        access_and_gather(dh, domain_shape)
        synchronization(dh, test_gpu=False)
        synchronization(dh, test_gpu=True)


def test_kernel():
    for domain_shape in [(4, 5), (3, 4, 5)]:
        dh = create_data_handling(domain_size=domain_shape, periodicity=True)
        assert all(dh.periodicity)
        kernel_execution_jacobi(dh, Target.CPU)
        reduction(dh)

        try:
            import pycuda
            dh = create_data_handling(domain_size=domain_shape, periodicity=True)
            kernel_execution_jacobi(dh, Target.GPU)
        except ImportError:
            pass


@pytest.mark.parametrize('target', (Target.CPU, Target.GPU))
def test_kernel_param(target):
    for domain_shape in [(4, 5), (3, 4, 5)]:
        if target == Target.GPU:
            pytest.importorskip('pycuda')

        dh = create_data_handling(domain_size=domain_shape, periodicity=True, default_target=target)
        kernel_execution_jacobi(dh, target)
        reduction(dh)


def test_vtk_output():
    pytest.importorskip('pyevtk')
    for domain_shape in [(4, 5), (3, 4, 5)]:
        dh = create_data_handling(domain_size=domain_shape, periodicity=True)
        vtk_output(dh)


def test_add_arrays():
    domain_shape = (3, 4, 5)
    field_description = 'x, y(9)'

    dh = create_data_handling(domain_size=domain_shape, default_ghost_layers=0, default_layout='numpy')
    x_, y_ = dh.add_arrays(field_description)

    x, y = ps.fields(field_description + ': [3,4,5]')

    assert x_ == x
    assert y_ == y
    assert x == dh.fields['x']
    assert y == dh.fields['y']


def test_get_kwarg():
    domain_shape = (10, 10)
    field_description = 'src, dst'

    dh = create_data_handling(domain_size=domain_shape, default_ghost_layers=1)
    src, dst = dh.add_arrays(field_description)
    dh.fill("src", 1.0, ghost_layers=True)
    dh.fill("dst", 0.0, ghost_layers=True)

    with pytest.raises(ValueError):
        dh.add_array('src')

    ur = ps.Assignment(src.center, dst.center)
    kernel = ps.create_kernel(ur).compile()

    kw = dh.get_kernel_kwargs(kernel)
    assert np.all(kw[0]['src'] == dh.cpu_arrays['src'])
    assert np.all(kw[0]['dst'] == dh.cpu_arrays['dst'])


def test_add_custom_data():
    pytest.importorskip('pycuda')

    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit  # noqa

    def cpu_data_create_func():
        return np.ones((2, 2), dtype=np.float64)

    def gpu_data_create_func():
        return gpuarray.zeros((2, 2), dtype=np.float64)

    def cpu_to_gpu_transfer_func(gpuarr, cpuarray):
        gpuarr.set(cpuarray)

    def gpu_to_cpu_transfer_func(gpuarr, cpuarray):
        gpuarr.get(cpuarray)

    dh = create_data_handling(domain_size=(10, 10))
    dh.add_custom_data('custom_data',
                       cpu_data_create_func,
                       gpu_data_create_func,
                       cpu_to_gpu_transfer_func,
                       gpu_to_cpu_transfer_func)

    assert np.all(dh.custom_data_cpu['custom_data'] == 1)
    assert np.all(dh.custom_data_gpu['custom_data'].get() == 0)

    dh.to_cpu(name='custom_data')
    dh.to_gpu(name='custom_data')

    assert 'custom_data' in dh.custom_data_names


def test_log():
    dh = create_data_handling(domain_size=(10, 10))
    dh.log_on_root()
    assert dh.is_root
    assert dh.world_rank == 0


def test_save_data():
    domain_shape = (2, 2)

    dh = create_data_handling(domain_size=domain_shape, default_ghost_layers=1)
    dh.add_array("src", values_per_cell=9)
    dh.fill("src", 1.0, ghost_layers=True)
    dh.add_array("dst", values_per_cell=9)
    dh.fill("dst", 1.0, ghost_layers=True)

    dh.save_all(str(INPUT_FOLDER) + '/datahandling_save_test')


def test_load_data():
    domain_shape = (2, 2)

    dh = create_data_handling(domain_size=domain_shape, default_ghost_layers=1)
    dh.add_array("src", values_per_cell=9)
    dh.fill("src", 0.0, ghost_layers=True)
    dh.add_array("dst", values_per_cell=9)
    dh.fill("dst", 0.0, ghost_layers=True)

    dh.load_all(str(INPUT_FOLDER) + '/datahandling_load_test')
    assert np.all(dh.cpu_arrays['src']) == 1
    assert np.all(dh.cpu_arrays['dst']) == 1

    domain_shape = (3, 3)

    dh = create_data_handling(domain_size=domain_shape, default_ghost_layers=1)
    dh.add_array("src", values_per_cell=9)
    dh.fill("src", 0.0, ghost_layers=True)
    dh.add_array("dst", values_per_cell=9)
    dh.fill("dst", 0.0, ghost_layers=True)
    dh.add_array("dst2", values_per_cell=9)
    dh.fill("dst2", 0.0, ghost_layers=True)

    dh.load_all(str(INPUT_FOLDER) + '/datahandling_load_test')
    assert np.all(dh.cpu_arrays['src']) == 0
    assert np.all(dh.cpu_arrays['dst']) == 0
    assert np.all(dh.cpu_arrays['dst2']) == 0


def test_array_handler():
    size = (2, 2)
    pytest.importorskip('pycuda')
    array_handler = PyCudaArrayHandler()

    zero_array = array_handler.zeros(size)
    cpu_array = np.empty(size)
    array_handler.download(zero_array, cpu_array)
    assert np.all(cpu_array) == 0

    ones_array = array_handler.ones(size)
    cpu_array = np.empty(size)
    array_handler.download(ones_array, cpu_array)
    assert np.all(cpu_array) == 1

    empty = array_handler.empty(size)
    assert empty.strides == (16, 8)
    empty = array_handler.empty(shape=size, layout=(1, 0))
    assert empty.strides == (8, 16)

    random_array = array_handler.randn(size)

