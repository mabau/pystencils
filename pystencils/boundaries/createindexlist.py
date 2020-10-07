import itertools
import warnings

import numpy as np

try:
    # Try to import right away - assume compiled code is available
    # compile with: python setup.py build_ext --inplace --use-cython
    from pystencils.boundaries.createindexlistcython import create_boundary_neighbor_index_list_2d, \
        create_boundary_neighbor_index_list_3d, create_boundary_cell_index_list_2d, create_boundary_cell_index_list_3d

    cython_funcs_available = True
except ImportError:
    try:
        # If not, try development mode and import via pyximport
        import pyximport

        pyximport.install(language_level=3)
        cython_funcs_available = True
    except ImportError:
        cython_funcs_available = False
    if cython_funcs_available:
        from pystencils.boundaries.createindexlistcython import create_boundary_neighbor_index_list_2d, \
            create_boundary_neighbor_index_list_3d, create_boundary_cell_index_list_2d, \
            create_boundary_cell_index_list_3d

boundary_index_array_coordinate_names = ["x", "y", "z"]
direction_member_name = "dir"


def numpy_data_type_for_boundary_object(boundary_object, dim):
    coordinate_names = boundary_index_array_coordinate_names[:dim]
    return np.dtype([(name, np.int32) for name in coordinate_names]
                    + [(direction_member_name, np.int32)]
                    + [(i[0], i[1].numpy_dtype) for i in boundary_object.additional_data], align=True)


def _create_boundary_neighbor_index_list_python(flag_field_arr, nr_of_ghost_layers, boundary_mask,
                                                fluid_mask, stencil, single_link):
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
                if single_link:
                    break

    return np.array(result, dtype=index_arr_dtype)


def _create_boundary_cell_index_list_python(flag_field_arr, boundary_mask,
                                            fluid_mask, stencil, single_link):
    coordinate_names = boundary_index_array_coordinate_names[:len(flag_field_arr.shape)]
    index_arr_dtype = np.dtype([(name, np.int32) for name in coordinate_names] + [(direction_member_name, np.int32)])

    result = []
    for cell in itertools.product(*reversed([range(0, i) for i in flag_field_arr.shape])):
        cell = cell[::-1]
        if not flag_field_arr[cell] & boundary_mask:
            continue
        for dir_idx, direction in enumerate(stencil):
            neighbor_cell = tuple([cell_i + dir_i for cell_i, dir_i in zip(cell, direction)])
            if any(not 0 <= e < upper for e, upper in zip(neighbor_cell, flag_field_arr.shape)):
                continue
            if flag_field_arr[neighbor_cell] & fluid_mask:
                result.append(cell + (dir_idx,))
                if single_link:
                    break

    return np.array(result, dtype=index_arr_dtype)


def create_boundary_index_list(flag_field, stencil, boundary_mask, fluid_mask,
                               nr_of_ghost_layers=1, inner_or_boundary=True, single_link=False):
    """Creates a numpy array storing links (connections) between domain cells and boundary cells.

    Args:
        flag_field: flag integer array where boundary and domain cells are marked (interpreted as bit vector)
        stencil: list of directions, for possible links. When single_link is set to true the order matters, because
                 then only the first link is added to the list
        boundary_mask: cells where (cell & mask) is true are considered boundary cells
        fluid_mask: cells where (cell & mask) is true are considered fluid/inner cells cells
        nr_of_ghost_layers: only relevant if neighbors is True
        inner_or_boundary: if true, the result contains the cell coordinates of the domain cells -
                    if false the boundary cells are listed
        single_link: if true only the first link is reported from this cell

    """
    dim = len(flag_field.shape)
    coordinate_names = boundary_index_array_coordinate_names[:dim]
    index_arr_dtype = np.dtype([(name, np.int32) for name in coordinate_names] + [(direction_member_name, np.int32)])

    stencil = np.array(stencil, dtype=np.int32)
    args = (flag_field, nr_of_ghost_layers, boundary_mask, fluid_mask, stencil, single_link)
    args_no_gl = (flag_field, boundary_mask, fluid_mask, stencil, single_link)

    if cython_funcs_available:
        if dim == 2:
            if inner_or_boundary:
                idx_list = create_boundary_neighbor_index_list_2d(*args)
            else:
                idx_list = create_boundary_cell_index_list_2d(*args_no_gl)
        elif dim == 3:
            if inner_or_boundary:
                idx_list = create_boundary_neighbor_index_list_3d(*args)
            else:
                idx_list = create_boundary_cell_index_list_3d(*args_no_gl)
        else:
            raise ValueError("Flag field has to be a 2 or 3 dimensional numpy array")
        return np.array(idx_list, dtype=index_arr_dtype)
    else:
        if flag_field.size > 1e6:
            warnings.warn("Boundary setup may take very long! Consider installing cython to speed it up")
        if inner_or_boundary:
            return _create_boundary_neighbor_index_list_python(*args)
        else:
            return _create_boundary_cell_index_list_python(*args_no_gl)


def create_boundary_index_array(flag_field, stencil, boundary_mask, fluid_mask, boundary_object,
                                nr_of_ghost_layers=1, inner_or_boundary=True, single_link=False):
    idx_array = create_boundary_index_list(flag_field, stencil, boundary_mask, fluid_mask,
                                           nr_of_ghost_layers, inner_or_boundary, single_link)
    dim = len(flag_field.shape)

    if boundary_object.additional_data:
        coordinate_names = boundary_index_array_coordinate_names[:dim]
        index_arr_dtype = numpy_data_type_for_boundary_object(boundary_object, dim)
        extended_idx_field = np.empty(len(idx_array), dtype=index_arr_dtype)
        for prop in coordinate_names + ['dir']:
            extended_idx_field[prop] = idx_array[prop]

        idx_array = extended_idx_field

    return idx_array
