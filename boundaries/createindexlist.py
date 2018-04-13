import numpy as np
import itertools
import warnings

try:
    import pyximport

    pyximport.install()
    from pystencils.boundaries.createindexlistcython import create_boundary_index_list_2d, create_boundary_index_list_3d

    cython_funcs_available = True
except Exception:
    cython_funcs_available = False
    create_boundary_index_list_2d = None
    create_boundary_index_list_3d = None

boundary_index_array_coordinate_names = ["x", "y", "z"]
direction_member_name = "dir"


def numpy_data_type_for_boundary_object(boundary_object, dim):
    coordinate_names = boundary_index_array_coordinate_names[:dim]
    return np.dtype([(name, np.int32) for name in coordinate_names] +
                    [(direction_member_name, np.int32)] +
                    [(i[0], i[1].numpy_dtype) for i in boundary_object.additional_data], align=True)


def _create_boundary_index_list_python(flag_field_arr, nr_of_ghost_layers, boundary_mask, fluid_mask, stencil):
    coordinate_names = boundary_index_array_coordinate_names[:len(flag_field_arr.shape)]
    index_arr_dtype = np.dtype([(name, np.int32) for name in coordinate_names] + [(direction_member_name, np.int32)])

    result = []
    gl = nr_of_ghost_layers
    for cell in itertools.product(*reversed([range(gl, i - gl) for i in flag_field_arr.shape])):
        cell = cell[::-1]
        if not flag_field_arr[cell] & fluid_mask:
            continue
        for dir_idx, direction in enumerate(stencil):
            neighbor_cell = tuple([cell_i + dir_i for cell_i, dir_i in zip(cell, direction)])
            if flag_field_arr[neighbor_cell] & boundary_mask:
                result.append(cell + (dir_idx,))

    return np.array(result, dtype=index_arr_dtype)


def create_boundary_index_list(flag_field, stencil, boundary_mask, fluid_mask, nr_of_ghost_layers=1):
    dim = len(flag_field.shape)
    coordinate_names = boundary_index_array_coordinate_names[:dim]
    index_arr_dtype = np.dtype([(name, np.int32) for name in coordinate_names] + [(direction_member_name, np.int32)])

    if cython_funcs_available:
        stencil = np.array(stencil, dtype=np.int32)
        if dim == 2:
            idx_list = create_boundary_index_list_2d(flag_field, nr_of_ghost_layers, boundary_mask, fluid_mask, stencil)
        elif dim == 3:
            idx_list = create_boundary_index_list_3d(flag_field, nr_of_ghost_layers, boundary_mask, fluid_mask, stencil)
        else:
            raise ValueError("Flag field has to be a 2 or 3 dimensional numpy array")
        return np.array(idx_list, dtype=index_arr_dtype)
    else:
        if flag_field.size > 1e6:
            warnings.warn("Boundary setup may take very long! Consider installing cython to speed it up")
        return _create_boundary_index_list_python(flag_field, nr_of_ghost_layers, boundary_mask, fluid_mask, stencil)


def create_boundary_index_array(flag_field, stencil, boundary_mask, fluid_mask, boundary_object, nr_of_ghost_layers=1):
    idx_array = create_boundary_index_list(flag_field, stencil, boundary_mask, fluid_mask, nr_of_ghost_layers)
    dim = len(flag_field.shape)

    if boundary_object.additional_data:
        coordinate_names = boundary_index_array_coordinate_names[:dim]
        index_arr_dtype = numpy_data_type_for_boundary_object(boundary_object, dim)
        extended_idx_field = np.empty(len(idx_array), dtype=index_arr_dtype)
        for prop in coordinate_names + ['dir']:
            extended_idx_field[prop] = idx_array[prop]

        idx_array = extended_idx_field

    return idx_array
