"""Tests for the (un)packing (from)to buffers on a CUDA GPU."""

from dataclasses import replace
import numpy as np
import pytest

import pystencils
from pystencils import Assignment, Field, FieldType, Target, CreateKernelConfig, create_kernel, fields
from pystencils.bit_masks import flag_cond
from pystencils.field import create_numpy_array_with_layout, layout_string_to_tuple
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

        config = CreateKernelConfig(target=pystencils.Target.GPU, data_type=pack_types)
        pack_ast = create_kernel(pack_eqs, config=config)

        pack_kernel = pack_ast.compile()
        pack_kernel(buffer=gpu_buffer_arr, src_field=gpu_src_arr)

        unpack_eqs = [Assignment(dst_field.center(), buffer.center())]
        unpack_types = {'dst_field': gpu_dst_arr.dtype, 'buffer': gpu_buffer_arr.dtype}

        config = CreateKernelConfig(target=pystencils.Target.GPU, data_type=unpack_types)
        unpack_ast = create_kernel(unpack_eqs, config=config)

        unpack_kernel = unpack_ast.compile()
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

            config = CreateKernelConfig(target=pystencils.Target.GPU, data_type=pack_types)
            pack_ast = create_kernel(pack_eqs, config=config)

            pack_kernel = pack_ast.compile()
            pack_kernel(buffer=gpu_buffer_arr, src_field=gpu_src_arr[pack_slice])

            # Unpack into ghost layer of dst_field in N direction
            unpack_eqs = [Assignment(dst_field.center(), buffer.center())]
            unpack_types = {'dst_field': gpu_dst_arr.dtype, 'buffer': gpu_buffer_arr.dtype}

            config = CreateKernelConfig(target=pystencils.Target.GPU, data_type=unpack_types)
            unpack_ast = create_kernel(unpack_eqs, config=config)

            unpack_kernel = unpack_ast.compile()
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

        config = CreateKernelConfig(target=pystencils.Target.GPU, data_type=pack_types)
        pack_code = create_kernel(pack_eqs, config=config)
        pack_kernel = pack_code.compile()

        pack_kernel(buffer=gpu_buffer_arr, src_field=gpu_src_arr)

        unpack_eqs = []

        for idx in range(num_cell_values):
            eq = Assignment(dst_field(idx), buffer(idx))
            unpack_eqs.append(eq)

        unpack_types = {'dst_field': gpu_dst_arr.dtype, 'buffer': gpu_buffer_arr.dtype}

        config = CreateKernelConfig(target=pystencils.Target.GPU, data_type=unpack_types)
        unpack_ast = create_kernel(unpack_eqs, config=config)
        unpack_kernel = unpack_ast.compile()
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
        config = CreateKernelConfig(target=pystencils.Target.GPU, data_type=pack_types)
        pack_ast = create_kernel(pack_eqs, config=config)
        pack_kernel = pack_ast.compile()
        pack_kernel(buffer=gpu_buffer_arr, src_field=gpu_src_arr)

        unpack_eqs = []

        for buffer_idx, cell_idx in enumerate(cell_indices):
            eq = Assignment(dst_field(cell_idx), buffer(buffer_idx))
            unpack_eqs.append(eq)

        unpack_types = {'dst_field': gpu_dst_arr.dtype, 'buffer': gpu_buffer_arr.dtype}
        config = CreateKernelConfig(target=pystencils.Target.GPU, data_type=unpack_types)
        unpack_ast = create_kernel(unpack_eqs, config=config)
        unpack_kernel = unpack_ast.compile()

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
            config = CreateKernelConfig(target=pystencils.Target.GPU, data_type=pack_types)
            pack_ast = create_kernel(pack_eqs, config=config)
            pack_kernel = pack_ast.compile()

            pack_kernel(buffer=gpu_buffer_arr, src_field=gpu_src_arr)

            unpack_eqs = []

            for idx in range(num_cell_values):
                eq = Assignment(dst_field(idx), buffer(idx))
                unpack_eqs.append(eq)

            unpack_types = {'dst_field': gpu_dst_arr.dtype, 'buffer': gpu_buffer_arr.dtype}
            config = CreateKernelConfig(target=pystencils.Target.GPU, data_type=unpack_types)
            unpack_ast = create_kernel(unpack_eqs, config=config)
            unpack_kernel = unpack_ast.compile()

            unpack_kernel(buffer=gpu_buffer_arr, dst_field=gpu_dst_arr)


def test_buffer_indexing():
    src_field, dst_field = fields(f'pdfs_src(19), pdfs_dst(19) :double[3D]')
    mask_field = fields(f'mask : uint32 [3D]')
    buffer = Field.create_generic('buffer', spatial_dimensions=1, field_type=FieldType.BUFFER,
                                  dtype="float64",
                                  index_shape=(19,))

    src_field_size = src_field.spatial_shape
    mask_field_size = mask_field.spatial_shape

    up = Assignment(buffer(0), flag_cond(1, mask_field.center, src_field[0, 1, 0](1)))
    iteration_slice = tuple(slice(None, None, 2) for _ in range(3))
    config = CreateKernelConfig(target=Target.GPU)
    config = replace(config, iteration_slice=iteration_slice, ghost_layers=0)

    ast = create_kernel(up, config=config)
    parameters = ast.get_parameters()

    spatial_shape_symbols = [p.symbol for p in parameters if p.is_field_shape]

    # The loop counters as well as the resolved field access should depend on one common spatial shape
    if spatial_shape_symbols[0] in mask_field_size:
        for s in spatial_shape_symbols:
            assert s in mask_field_size

    if spatial_shape_symbols[0] in src_field_size:
        for s in spatial_shape_symbols:
            assert s in src_field_size

    assert len(spatial_shape_symbols) <= 3
