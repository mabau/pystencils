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
default_index_array_dtype = np.int32


def numpy_data_type_for_boundary_object(boundary_object, dim):
    coordinate_names = boundary_index_array_coordinate_names[:dim]
    return np.dtype([(name, default_index_array_dtype) for name in coordinate_names]
                    + [(direction_member_name, default_index_array_dtype)]
                    + [(i[0], i[1].numpy_dtype) for i in boundary_object.additional_data], align=True)


def _create_index_list_python(flag_field_arr, boundary_mask,
                              fluid_mask, stencil, single_link, inner_or_boundary=False, nr_of_ghost_layers=None):

    if inner_or_boundary and nr_of_ghost_layers is None:
        raise ValueError("If inner_or_boundary is set True the number of ghost layers "
                         "around the inner domain has to be specified")

    if nr_of_ghost_layers is None:
        nr_of_ghost_layers = 0

    coordinate_names = boundary_index_array_coordinate_names[:len(flag_field_arr.shape)]
    index_arr_dtype = np.dtype([(name, default_index_array_dtype) for name in coordinate_names]
                               + [(direction_member_name, default_index_array_dtype)])

    # boundary cells are extracted via np.where. To ensure continous memory access in the compute kernel these cells
    # have to be sorted.
    boundary_cells = np.transpose(np.nonzero(flag_field_arr == boundary_mask))
    for i in range(len(flag_field_arr.shape)):
        boundary_cells = boundary_cells[boundary_cells[:, i].argsort(kind='mergesort')]

    # First a set is created to save all fluid cells which are near boundary
    fluid_cells = set()
    for cell in boundary_cells:
        cell = tuple(cell)
        for dir_idx, direction in enumerate(stencil):
            neighbor_cell = tuple([cell_i + dir_i for cell_i, dir_i in zip(cell, direction)])
            # prevent out ouf bounds access. If boundary cell is at the border, some stencil directions would be out.
            if any(not 0 + nr_of_ghost_layers <= e < upper - nr_of_ghost_layers
                   for e, upper in zip(neighbor_cell, flag_field_arr.shape)):
                continue
            if flag_field_arr[neighbor_cell] & fluid_mask:
                fluid_cells.add(neighbor_cell)

    # then this is set is transformed to a list to make it sortable. This ensures continoous memory access later.
    fluid_cells = list(fluid_cells)
    if len(flag_field_arr.shape) == 3:
        fluid_cells.sort(key=lambda tup: (tup[-1], tup[-2], tup[0]))
    else:
        fluid_cells.sort(key=lambda tup: (tup[-1], tup[0]))

    cells_to_iterate = fluid_cells if inner_or_boundary else boundary_cells
    checkmask = boundary_mask if inner_or_boundary else fluid_mask

    result = []
    for cell in cells_to_iterate:
        cell = tuple(cell)
        sum_cells = np.zeros(len(cell))
        for dir_idx, direction in enumerate(stencil):
            neighbor_cell = tuple([cell_i + dir_i for cell_i, dir_i in zip(cell, direction)])
            # prevent out ouf bounds access. If boundary cell is at the border, some stencil directions would be out.
            if any(not 0 <= e < upper for e, upper in zip(neighbor_cell, flag_field_arr.shape)):
                continue
            if flag_field_arr[neighbor_cell] & checkmask:
                if single_link:
                    sum_cells += np.array(direction)
                else:
                    result.append(tuple(cell) + (dir_idx,))

        # the discrete normal direction is the one which gives the maximum inner product to the stencil direction
        if single_link and any(sum_cells != 0):
            idx = np.argmax(np.inner(sum_cells, stencil))
            result.append(tuple(cell) + (idx,))

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
        single_link: if true only the link in normal direction to this cell is reported

    """
    dim = len(flag_field.shape)
    coordinate_names = boundary_index_array_coordinate_names[:dim]
    index_arr_dtype = np.dtype([(name, default_index_array_dtype) for name in coordinate_names]
                               + [(direction_member_name, default_index_array_dtype)])

    stencil = np.array(stencil, dtype=default_index_array_dtype)
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
        return _create_index_list_python(*args_no_gl, inner_or_boundary=inner_or_boundary,
                                         nr_of_ghost_layers=nr_of_ghost_layers)


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
