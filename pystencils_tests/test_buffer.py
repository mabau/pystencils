"""Tests  (un)packing (from)to buffers."""

import numpy as np

import pystencils as ps
from pystencils import Assignment, Field, FieldType, create_kernel
from pystencils.field import create_numpy_array_with_layout, layout_string_to_tuple
from pystencils.slicing import (
    add_ghost_layers, get_ghost_region_slice, get_slice_before_ghost_layer)
from pystencils.stencil import direction_string_to_offset

FIELD_SIZES = [(32, 10), (10, 8, 6)]


def _generate_fields(dt=np.uint64, num_directions=1, layout='numpy'):
    field_sizes = FIELD_SIZES
    if num_directions > 1:
        field_sizes = [s + (num_directions,) for s in field_sizes]

    fields = []
    for size in field_sizes:
        field_layout = layout_string_to_tuple(layout, len(size))
        src_arr = create_numpy_array_with_layout(size, field_layout, dtype=dt)

        array_data = np.reshape(np.arange(1, int(np.prod(size)+1)), size)
        # Use flat iterator to input data into the array
        src_arr.flat = add_ghost_layers(array_data, index_dimensions=1 if num_directions > 1 else 0).astype(dt).flat
        dst_arr = np.zeros(src_arr.shape, dtype=dt)
        buffer_arr = np.zeros(np.prod(src_arr.shape), dtype=dt)
        fields.append((src_arr, dst_arr, buffer_arr))
    return fields


def test_full_scalar_field():
    """Tests fully (un)packing a scalar field (from)to a buffer."""
    fields = _generate_fields()
    for (src_arr, dst_arr, buffer_arr) in fields:
        src_field = Field.create_from_numpy_array("src_field", src_arr)
        dst_field = Field.create_from_numpy_array("dst_field", dst_arr)
        buffer = Field.create_generic("buffer", spatial_dimensions=1,
                                      field_type=FieldType.BUFFER, dtype=src_arr.dtype)

        pack_eqs = [Assignment(buffer.center(), src_field.center())]
        pack_code = create_kernel(pack_eqs, data_type={'src_field': src_arr.dtype, 'buffer': buffer.dtype})
        code = ps.get_code_str(pack_code)
        ps.show_code(pack_code)

        pack_kernel = pack_code.compile()
        pack_kernel(buffer=buffer_arr, src_field=src_arr)

        unpack_eqs = [Assignment(dst_field.center(), buffer.center())]
        unpack_code = create_kernel(unpack_eqs, data_type={'dst_field': dst_arr.dtype, 'buffer': buffer.dtype})

        unpack_kernel = unpack_code.compile()
        unpack_kernel(dst_field=dst_arr, buffer=buffer_arr)

        np.testing.assert_equal(src_arr, dst_arr)


def test_field_slice():
    """Tests (un)packing slices of a scalar field (from)to a buffer."""
    fields = _generate_fields()
    for d in ['N', 'S', 'NW', 'SW', 'TNW', 'B']:
        for (src_arr, dst_arr, bufferArr) in fields:
            # Extract slice from N direction of the field
            slice_dir = direction_string_to_offset(d, dim=len(src_arr.shape))
            pack_slice = get_slice_before_ghost_layer(slice_dir)
            unpack_slice = get_ghost_region_slice(slice_dir)

            src_field = Field.create_from_numpy_array("src_field", src_arr[pack_slice])
            dst_field = Field.create_from_numpy_array("dst_field", dst_arr[unpack_slice])
            buffer = Field.create_generic("buffer", spatial_dimensions=1,
                                          field_type=FieldType.BUFFER, dtype=src_arr.dtype)

            pack_eqs = [Assignment(buffer.center(), src_field.center())]
            pack_code = create_kernel(pack_eqs, data_type={'src_field': src_arr.dtype, 'buffer': buffer.dtype})

            pack_kernel = pack_code.compile()
            pack_kernel(buffer=bufferArr, src_field=src_arr[pack_slice])

            # Unpack into ghost layer of dst_field in N direction
            unpack_eqs = [Assignment(dst_field.center(), buffer.center())]
            unpack_code = create_kernel(unpack_eqs, data_type={'dst_field': dst_arr.dtype, 'buffer': buffer.dtype})

            unpack_kernel = unpack_code.compile()
            unpack_kernel(buffer=bufferArr, dst_field=dst_arr[unpack_slice])

            np.testing.assert_equal(src_arr[pack_slice], dst_arr[unpack_slice])


def test_all_cell_values():
    """Tests (un)packing all cell values of the a field (from)to a buffer."""
    num_cell_values = 19
    fields = _generate_fields(num_directions=num_cell_values)
    for (src_arr, dst_arr, bufferArr) in fields:
        src_field = Field.create_from_numpy_array("src_field", src_arr, index_dimensions=1)
        dst_field = Field.create_from_numpy_array("dst_field", dst_arr, index_dimensions=1)
        buffer = Field.create_generic("buffer", spatial_dimensions=1, index_dimensions=1,
                                      field_type=FieldType.BUFFER, dtype=src_arr.dtype)

        pack_eqs = []
        # Since we are packing all cell values for all cells, then
        # the buffer index is equivalent to the field index
        for idx in range(num_cell_values):
            eq = Assignment(buffer(idx), src_field(idx))
            pack_eqs.append(eq)

        pack_code = create_kernel(pack_eqs, data_type={'src_field': src_arr.dtype, 'buffer': buffer.dtype})
        pack_kernel = pack_code.compile()
        pack_kernel(buffer=bufferArr, src_field=src_arr)

        unpack_eqs = []

        for idx in range(num_cell_values):
            eq = Assignment(dst_field(idx), buffer(idx))
            unpack_eqs.append(eq)

        unpack_code = create_kernel(unpack_eqs, data_type={'dst_field': dst_arr.dtype, 'buffer': buffer.dtype})
        unpack_kernel = unpack_code.compile()
        unpack_kernel(buffer=bufferArr, dst_field=dst_arr)

        np.testing.assert_equal(src_arr, dst_arr)


def test_subset_cell_values():
    """Tests (un)packing a subset of cell values of the a field (from)to a buffer."""
    num_cell_values = 19
    # Cell indices of the field to be (un)packed (from)to the buffer
    cell_indices = [1, 5, 7, 8, 10, 12, 13]
    fields = _generate_fields(num_directions=num_cell_values)
    for (src_arr, dst_arr, bufferArr) in fields:
        src_field = Field.create_from_numpy_array("src_field", src_arr, index_dimensions=1)
        dst_field = Field.create_from_numpy_array("dst_field", dst_arr, index_dimensions=1)
        buffer = Field.create_generic("buffer", spatial_dimensions=1, index_dimensions=1,
                                      field_type=FieldType.BUFFER, dtype=src_arr.dtype)

        pack_eqs = []
        # Since we are packing all cell values for all cells, then
        # the buffer index is equivalent to the field index
        for buffer_idx, cell_idx in enumerate(cell_indices):
            eq = Assignment(buffer(buffer_idx), src_field(cell_idx))
            pack_eqs.append(eq)

        pack_code = create_kernel(pack_eqs, data_type={'src_field': src_arr.dtype, 'buffer': buffer.dtype})
        pack_kernel = pack_code.compile()
        pack_kernel(buffer=bufferArr, src_field=src_arr)

        unpack_eqs = []

        for buffer_idx, cell_idx in enumerate(cell_indices):
            eq = Assignment(dst_field(cell_idx), buffer(buffer_idx))
            unpack_eqs.append(eq)

        unpack_code = create_kernel(unpack_eqs, data_type={'dst_field': dst_arr.dtype, 'buffer': buffer.dtype})
        unpack_kernel = unpack_code.compile()
        unpack_kernel(buffer=bufferArr, dst_field=dst_arr)

        mask_arr = np.ma.masked_where((src_arr - dst_arr) != 0, src_arr)
        np.testing.assert_equal(dst_arr, mask_arr.filled(int(0)))


def test_field_layouts():
    num_cell_values = 27
    for layout_str in ['numpy', 'fzyx', 'zyxf', 'reverse_numpy']:
        fields = _generate_fields(num_directions=num_cell_values, layout=layout_str)
        for (src_arr, dst_arr, bufferArr) in fields:
            src_field = Field.create_from_numpy_array("src_field", src_arr, index_dimensions=1)
            dst_field = Field.create_from_numpy_array("dst_field", dst_arr, index_dimensions=1)
            buffer = Field.create_generic("buffer", spatial_dimensions=1, index_dimensions=1,
                                          field_type=FieldType.BUFFER, dtype=src_arr.dtype)

            pack_eqs = []
            # Since we are packing all cell values for all cells, then
            # the buffer index is equivalent to the field index
            for idx in range(num_cell_values):
                eq = Assignment(buffer(idx), src_field(idx))
                pack_eqs.append(eq)

            pack_code = create_kernel(pack_eqs, data_type={'src_field': src_arr.dtype, 'buffer': buffer.dtype})
            pack_kernel = pack_code.compile()
            pack_kernel(buffer=bufferArr, src_field=src_arr)

            unpack_eqs = []

            for idx in range(num_cell_values):
                eq = Assignment(dst_field(idx), buffer(idx))
                unpack_eqs.append(eq)

            unpack_code = create_kernel(unpack_eqs, data_type={'dst_field': dst_arr.dtype, 'buffer': buffer.dtype})
            unpack_kernel = unpack_code.compile()
            unpack_kernel(buffer=bufferArr, dst_field=dst_arr)


def test_iteration_slices():
    num_cell_values = 19
    dt = np.uint64
    fields = _generate_fields(dt=dt, num_directions=num_cell_values)
    for (src_arr, dst_arr, bufferArr) in fields:
        spatial_dimensions = len(src_arr.shape) - 1
        # src_field = Field.create_from_numpy_array("src_field", src_arr, index_dimensions=1)
        # dst_field = Field.create_from_numpy_array("dst_field", dst_arr, index_dimensions=1)
        src_field = Field.create_generic("src_field", spatial_dimensions, index_shape=(num_cell_values,), dtype=dt)
        dst_field = Field.create_generic("dst_field", spatial_dimensions, index_shape=(num_cell_values,), dtype=dt)
        buffer = Field.create_generic("buffer", spatial_dimensions=1, index_dimensions=1,
                                        field_type=FieldType.BUFFER, dtype=src_arr.dtype)

        pack_eqs = []
        # Since we are packing all cell values for all cells, then
        # the buffer index is equivalent to the field index
        for idx in range(num_cell_values):
            eq = Assignment(buffer(idx), src_field(idx))
            pack_eqs.append(eq)

        dim = src_field.spatial_dimensions

        #   Pack only the leftmost slice, only every second cell
        pack_slice = (slice(None, None, 2),) * (dim-1) + (0, )

        #   Fill the entire array with data
        src_arr[(slice(None, None, 1),) * dim] = np.arange(num_cell_values)
        dst_arr.fill(0)

        pack_code = create_kernel(pack_eqs, iteration_slice=pack_slice, data_type={'src_field': src_arr.dtype, 'buffer': buffer.dtype})
        pack_kernel = pack_code.compile()
        pack_kernel(buffer=bufferArr, src_field=src_arr)

        unpack_eqs = []

        for idx in range(num_cell_values):
            eq = Assignment(dst_field(idx), buffer(idx))
            unpack_eqs.append(eq)

        unpack_code = create_kernel(unpack_eqs, iteration_slice=pack_slice, data_type={'dst_field': dst_arr.dtype, 'buffer': buffer.dtype})
        unpack_kernel = unpack_code.compile()
        unpack_kernel(buffer=bufferArr, dst_field=dst_arr)

        #   Check if only every second entry of the leftmost slice has been copied
        np.testing.assert_equal(dst_arr[pack_slice], src_arr[pack_slice])
        np.testing.assert_equal(dst_arr[(slice(1, None, 2),)  * (dim-1) + (0,)], 0)
        np.testing.assert_equal(dst_arr[(slice(None, None, 1),)  * (dim-1) + (slice(1,None),)], 0)

