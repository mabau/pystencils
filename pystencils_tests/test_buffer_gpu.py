"""Tests for the (un)packing (from)to buffers on a CUDA GPU."""

import numpy as np
import pytest

from pystencils import Assignment, Field, FieldType
from pystencils.field import create_numpy_array_with_layout, layout_string_to_tuple
from pystencils.gpucuda import create_cuda_kernel, make_python_function
from pystencils.slicing import (
    add_ghost_layers, get_ghost_region_slice, get_slice_before_ghost_layer)
from pystencils.stencil import direction_string_to_offset

try:
    # noinspection PyUnresolvedReferences
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
except ImportError:
    pass


FIELD_SIZES = [(4, 3), (9, 3, 7)]


def _generate_fields(dt=np.uint8, stencil_directions=1, layout='numpy'):
    pytest.importorskip('pycuda')
    field_sizes = FIELD_SIZES
    if stencil_directions > 1:
        field_sizes = [s + (stencil_directions,) for s in field_sizes]

    fields = []
    for size in field_sizes:
        field_layout = layout_string_to_tuple(layout, len(size))
        src_arr = create_numpy_array_with_layout(size, field_layout).astype(dt)

        array_data = np.reshape(np.arange(1, int(np.prod(size)+1)), size)
        # Use flat iterator to input data into the array
        src_arr.flat = add_ghost_layers(array_data,
                                        index_dimensions=1 if stencil_directions > 1 else 0).astype(dt).flat

        gpu_src_arr = gpuarray.to_gpu(src_arr)
        gpu_dst_arr = gpuarray.empty_like(gpu_src_arr)
        size = int(np.prod(src_arr.shape))
        gpu_buffer_arr = gpuarray.zeros(size, dtype=dt)

        fields.append((src_arr, gpu_src_arr, gpu_dst_arr, gpu_buffer_arr))
    return fields


def test_full_scalar_field():
    """Tests fully (un)packing a scalar field (from)to a GPU buffer."""
    fields = _generate_fields()
    for (src_arr, gpu_src_arr, gpu_dst_arr, gpu_buffer_arr) in fields:
        src_field = Field.create_from_numpy_array("src_field", src_arr)
        dst_field = Field.create_from_numpy_array("dst_field", src_arr)
        buffer = Field.create_generic("buffer", spatial_dimensions=1,
                                      field_type=FieldType.BUFFER, dtype=src_arr.dtype)

        pack_eqs = [Assignment(buffer.center(), src_field.center())]
        pack_types = {'src_field': gpu_src_arr.dtype, 'buffer': gpu_buffer_arr.dtype}
        pack_code = create_cuda_kernel(pack_eqs, type_info=pack_types)

        pack_kernel = make_python_function(pack_code)
        pack_kernel(buffer=gpu_buffer_arr, src_field=gpu_src_arr)

        unpack_eqs = [Assignment(dst_field.center(), buffer.center())]
        unpack_types = {'dst_field': gpu_dst_arr.dtype, 'buffer': gpu_buffer_arr.dtype}
        unpack_code = create_cuda_kernel(unpack_eqs, type_info=unpack_types)

        unpack_kernel = make_python_function(unpack_code)
        unpack_kernel(dst_field=gpu_dst_arr, buffer=gpu_buffer_arr)

        dst_arr = gpu_dst_arr.get()

        np.testing.assert_equal(src_arr, dst_arr)


def test_field_slice():
    """Tests (un)packing slices of a scalar field (from)to a buffer."""
    fields = _generate_fields()
    for d in ['N', 'S', 'NW', 'SW', 'TNW', 'B']:
        for (src_arr, gpu_src_arr, gpu_dst_arr, gpu_buffer_arr) in fields:
            # Extract slice from N direction of the field
            slice_dir = direction_string_to_offset(d, dim=len(src_arr.shape))
            pack_slice = get_slice_before_ghost_layer(slice_dir)
            unpack_slice = get_ghost_region_slice(slice_dir)

            src_field = Field.create_from_numpy_array("src_field", src_arr[pack_slice])
            dst_field = Field.create_from_numpy_array("dst_field", src_arr[unpack_slice])
            buffer = Field.create_generic("buffer", spatial_dimensions=1,
                                          field_type=FieldType.BUFFER, dtype=src_arr.dtype)

            pack_eqs = [Assignment(buffer.center(), src_field.center())]
            pack_types = {'src_field': gpu_src_arr.dtype, 'buffer': gpu_buffer_arr.dtype}
            pack_code = create_cuda_kernel(pack_eqs, type_info=pack_types)

            pack_kernel = make_python_function(pack_code)
            pack_kernel(buffer=gpu_buffer_arr, src_field=gpu_src_arr[pack_slice])

            # Unpack into ghost layer of dst_field in N direction
            unpack_eqs = [Assignment(dst_field.center(), buffer.center())]
            unpack_types = {'dst_field': gpu_dst_arr.dtype, 'buffer': gpu_buffer_arr.dtype}
            unpack_code = create_cuda_kernel(unpack_eqs, type_info=unpack_types)

            unpack_kernel = make_python_function(unpack_code)
            unpack_kernel(buffer=gpu_buffer_arr, dst_field=gpu_dst_arr[unpack_slice])

            dst_arr = gpu_dst_arr.get()

            np.testing.assert_equal(src_arr[pack_slice], dst_arr[unpack_slice])


def test_all_cell_values():
    """Tests (un)packing all cell values of the a field (from)to a buffer."""
    num_cell_values = 7
    fields = _generate_fields(stencil_directions=num_cell_values)
    for (src_arr, gpu_src_arr, gpu_dst_arr, gpu_buffer_arr) in fields:
        src_field = Field.create_from_numpy_array("src_field", gpu_src_arr, index_dimensions=1)
        dst_field = Field.create_from_numpy_array("dst_field", gpu_src_arr, index_dimensions=1)
        buffer = Field.create_generic("buffer", spatial_dimensions=1, index_dimensions=1,
                                      field_type=FieldType.BUFFER, dtype=gpu_src_arr.dtype)

        pack_eqs = []
        # Since we are packing all cell values for all cells, then
        # the buffer index is equivalent to the field index
        for idx in range(num_cell_values):
            eq = Assignment(buffer(idx), src_field(idx))
            pack_eqs.append(eq)

        pack_types = {'src_field': gpu_src_arr.dtype, 'buffer': gpu_buffer_arr.dtype}
        pack_code = create_cuda_kernel(pack_eqs, type_info=pack_types)
        pack_kernel = make_python_function(pack_code)
        pack_kernel(buffer=gpu_buffer_arr, src_field=gpu_src_arr)

        unpack_eqs = []

        for idx in range(num_cell_values):
            eq = Assignment(dst_field(idx), buffer(idx))
            unpack_eqs.append(eq)

        unpack_types = {'dst_field': gpu_dst_arr.dtype, 'buffer': gpu_buffer_arr.dtype}
        unpack_code = create_cuda_kernel(unpack_eqs, type_info=unpack_types)
        unpack_kernel = make_python_function(unpack_code)
        unpack_kernel(buffer=gpu_buffer_arr, dst_field=gpu_dst_arr)

        dst_arr = gpu_dst_arr.get()

        np.testing.assert_equal(src_arr, dst_arr)


def test_subset_cell_values():
    """Tests (un)packing a subset of cell values of the a field (from)to a buffer."""
    num_cell_values = 7
    # Cell indices of the field to be (un)packed (from)to the buffer
    cell_indices = [1, 3, 5, 6]
    fields = _generate_fields(stencil_directions=num_cell_values)
    for (src_arr, gpu_src_arr, gpu_dst_arr, gpu_buffer_arr) in fields:
        src_field = Field.create_from_numpy_array("src_field", gpu_src_arr, index_dimensions=1)
        dst_field = Field.create_from_numpy_array("dst_field", gpu_src_arr, index_dimensions=1)
        buffer = Field.create_generic("buffer", spatial_dimensions=1, index_dimensions=1,
                                      field_type=FieldType.BUFFER, dtype=gpu_src_arr.dtype)

        pack_eqs = []
        # Since we are packing all cell values for all cells, then
        # the buffer index is equivalent to the field index
        for buffer_idx, cell_idx in enumerate(cell_indices):
            eq = Assignment(buffer(buffer_idx), src_field(cell_idx))
            pack_eqs.append(eq)

        pack_types = {'src_field': gpu_src_arr.dtype, 'buffer': gpu_buffer_arr.dtype}
        pack_code = create_cuda_kernel(pack_eqs, type_info=pack_types)
        pack_kernel = make_python_function(pack_code)
        pack_kernel(buffer=gpu_buffer_arr, src_field=gpu_src_arr)

        unpack_eqs = []

        for buffer_idx, cell_idx in enumerate(cell_indices):
            eq = Assignment(dst_field(cell_idx), buffer(buffer_idx))
            unpack_eqs.append(eq)

        unpack_types = {'dst_field': gpu_dst_arr.dtype, 'buffer': gpu_buffer_arr.dtype}
        unpack_code = create_cuda_kernel(unpack_eqs, type_info=unpack_types)
        unpack_kernel = make_python_function(unpack_code)
        unpack_kernel(buffer=gpu_buffer_arr, dst_field=gpu_dst_arr)

        dst_arr = gpu_dst_arr.get()

        mask_arr = np.ma.masked_where((src_arr - dst_arr) != 0, src_arr)
        np.testing.assert_equal(dst_arr, mask_arr.filled(int(0)))


def test_field_layouts():
    num_cell_values = 7
    for layout_str in ['numpy', 'fzyx', 'zyxf', 'reverse_numpy']:
        fields = _generate_fields(stencil_directions=num_cell_values, layout=layout_str)
        for (src_arr, gpu_src_arr, gpu_dst_arr, gpu_buffer_arr) in fields:
            src_field = Field.create_from_numpy_array("src_field", gpu_src_arr, index_dimensions=1)
            dst_field = Field.create_from_numpy_array("dst_field", gpu_src_arr, index_dimensions=1)
            buffer = Field.create_generic("buffer", spatial_dimensions=1, index_dimensions=1,
                                          field_type=FieldType.BUFFER, dtype=src_arr.dtype)

            pack_eqs = []
            # Since we are packing all cell values for all cells, then
            # the buffer index is equivalent to the field index
            for idx in range(num_cell_values):
                eq = Assignment(buffer(idx), src_field(idx))
                pack_eqs.append(eq)

            pack_types = {'src_field': gpu_src_arr.dtype, 'buffer': gpu_buffer_arr.dtype}
            pack_code = create_cuda_kernel(pack_eqs, type_info=pack_types)
            pack_kernel = make_python_function(pack_code)
            pack_kernel(buffer=gpu_buffer_arr, src_field=gpu_src_arr)

            unpack_eqs = []

            for idx in range(num_cell_values):
                eq = Assignment(dst_field(idx), buffer(idx))
                unpack_eqs.append(eq)

            unpack_types = {'dst_field': gpu_dst_arr.dtype, 'buffer': gpu_buffer_arr.dtype}
            unpack_code = create_cuda_kernel(unpack_eqs, type_info=unpack_types)
            unpack_kernel = make_python_function(unpack_code)
            unpack_kernel(buffer=gpu_buffer_arr, dst_field=gpu_dst_arr)
