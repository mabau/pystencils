import sympy as sp

from pystencils.field import create_numpy_array_with_layout, get_layout_of_array


class SliceMaker(object):
    def __getitem__(self, item):
        return item


make_slice = SliceMaker()


class SlicedGetter(object):
    def __init__(self, function_returning_array):
        self._functionReturningArray = function_returning_array

    def __getitem__(self, item):
        return self._functionReturningArray(item)


class SlicedGetterDataHandling:
    def __init__(self, data_handling, name):
        self.dh = data_handling
        self.name = name

    def __getitem__(self, slice_obj):
        if slice_obj is None:
            slice_obj = make_slice[:, :] if self.data_handling.dim == 2 else make_slice[:, :, 0.5]
        return self.dh.gather_array(self.name, slice_obj).squeeze()


def normalize_slice(slices, sizes):
    """Converts slices with floating point and/or negative entries to integer slices"""

    if len(slices) != len(sizes):
        raise ValueError("Slice dimension does not match sizes")

    result = []

    for s, size in zip(slices, sizes):
        if type(s) is int:
            if s < 0:
                s = size + s
            result.append(s)
            continue
        if type(s) is float:
            result.append(int(s * size))
            continue

        assert (type(s) is slice)

        if s.start is None:
            new_start = 0
        elif type(s.start) is float:
            new_start = int(s.start * size)
        elif not isinstance(s.start, sp.Basic) and s.start < 0:
            new_start = size + s.start
        else:
            new_start = s.start

        if s.stop is None:
            new_stop = size
        elif type(s.stop) is float:
            new_stop = int(s.stop * size)
        elif not isinstance(s.stop, sp.Basic) and s.stop < 0:
            new_stop = size + s.stop
        else:
            new_stop = s.stop

        result.append(slice(new_start, new_stop, s.step if s.step is not None else 1))

    return tuple(result)


def shift_slice(slices, offset):
    def shift_slice_component(slice_comp, shift_offset):
        if slice_comp is None:
            return None
        elif isinstance(slice_comp, int):
            return slice_comp + shift_offset
        elif isinstance(slice_comp, float):
            return slice_comp  # relative entries are not shifted
        elif isinstance(slice_comp, slice):
            return slice(shift_slice_component(slice_comp.start, shift_offset),
                         shift_slice_component(slice_comp.stop, shift_offset),
                         slice_comp.step)
        else:
            raise ValueError()

    if hasattr(offset, '__len__'):
        return tuple(shift_slice_component(k, off) for k, off in zip(slices, offset))
    else:
        if isinstance(slices, slice) or isinstance(slices, int) or isinstance(slices, float):
            return shift_slice_component(slices, offset)
        else:
            return tuple(shift_slice_component(k, offset) for k in slices)


def slice_from_direction(direction_name, dim, normal_offset=0, tangential_offset=0):
    """
    Create a slice from a direction named by compass scheme:
        i.e. 'N' for north returns same as make_slice[:, -1]
        the naming is:
            - x: W, E (west, east)
            - y: S, N (south, north)
            - z: B, T (bottom, top)
    Also combinations are allowed like north-east 'NE'

    :param direction_name: name of direction as explained above
    :param dim: dimension of the returned slice (should be 2 or 3)
    :param normal_offset: the offset in 'normal' direction: e.g. slice_from_direction('N',2, normal_offset=2)
                         would return make_slice[:, -3]
    :param tangential_offset: offset in the other directions: e.g. slice_from_direction('N',2, tangential_offset=2)
                         would return make_slice[2:-2, -1]
    """
    if tangential_offset == 0:
        result = [slice(None, None, None)] * dim
    else:
        result = [slice(tangential_offset, -tangential_offset, None)] * dim

    normal_slice_high, normal_slice_low = -1 - normal_offset, normal_offset

    for dim_idx, (low_name, high_name) in enumerate([('W', 'E'), ('S', 'N'), ('B', 'T')]):
        if low_name in direction_name:
            assert high_name not in direction_name, "Invalid direction name"
            result[dim_idx] = normal_slice_low
        if high_name in direction_name:
            assert low_name not in direction_name, "Invalid direction name"
            result[dim_idx] = normal_slice_high
    return tuple(result)


def remove_ghost_layers(arr, index_dimensions=0, ghost_layers=1):
    if ghost_layers <= 0:
        return arr
    dimensions = len(arr.shape)
    spatial_dimensions = dimensions - index_dimensions
    indexing = [slice(ghost_layers, -ghost_layers, None), ] * spatial_dimensions
    indexing += [slice(None, None, None)] * index_dimensions
    return arr[tuple(indexing)]


def add_ghost_layers(arr, index_dimensions=0, ghost_layers=1, layout=None):
    dimensions = len(arr.shape)
    spatial_dimensions = dimensions - index_dimensions
    new_shape = [e + 2 * ghost_layers for e in arr.shape[:spatial_dimensions]] + list(arr.shape[spatial_dimensions:])
    if layout is None:
        layout = get_layout_of_array(arr)
    result = create_numpy_array_with_layout(new_shape, layout)
    result.fill(0.0)
    indexing = [slice(ghost_layers, -ghost_layers, None), ] * spatial_dimensions
    indexing += [slice(None, None, None)] * index_dimensions
    result[tuple(indexing)] = arr
    return result


def get_slice_before_ghost_layer(direction, ghost_layers=1, thickness=None, full_slice=False):
    """
    Returns slicing expression for region before ghost layer
    :param direction: tuple specifying direction of slice
    :param ghost_layers: number of ghost layers
    :param thickness: thickness of the slice, defaults to number of ghost layers
    :param full_slice:  if true also the ghost cells in directions orthogonal to direction are contained in the
                       returned slice. Example (d=W ): if full_slice then also the ghost layer in N-S and T-B
                       are included, otherwise only inner cells are returned
    """
    if not thickness:
        thickness = ghost_layers
    full_slice_inc = ghost_layers if not full_slice else 0
    slices = []
    for dir_component in direction:
        if dir_component == -1:
            s = slice(ghost_layers, thickness + ghost_layers)
        elif dir_component == 0:
            end = -full_slice_inc
            s = slice(full_slice_inc, end if end != 0 else None)
        elif dir_component == 1:
            start = -thickness - ghost_layers
            end = -ghost_layers
            s = slice(start if start != 0 else None, end if end != 0 else None)
        else:
            raise ValueError("Invalid direction: only -1, 0, 1 components are allowed")
        slices.append(s)
    return tuple(slices)


def get_ghost_region_slice(direction, ghost_layers=1, thickness=None, full_slice=False):
    """
    Returns slice of ghost region. For parameters see :func:`get_slice_before_ghost_layer`
    """
    if not thickness:
        thickness = ghost_layers
    assert thickness > 0
    assert thickness <= ghost_layers
    full_slice_inc = ghost_layers if not full_slice else 0
    slices = []
    for dir_component in direction:
        if dir_component == -1:
            s = slice(ghost_layers - thickness, ghost_layers)
        elif dir_component == 0:
            end = -full_slice_inc
            s = slice(full_slice_inc, end if end != 0 else None)
        elif dir_component == 1:
            start = -ghost_layers
            end = - ghost_layers + thickness
            s = slice(start if start != 0 else None, end if end != 0 else None)
        else:
            raise ValueError("Invalid direction: only -1, 0, 1 components are allowed")
        slices.append(s)
    return tuple(slices)


def get_periodic_boundary_src_dst_slices(stencil, ghost_layers=1, thickness=None):
    src_dst_slice_tuples = []

    for d in stencil:
        if sum([abs(e) for e in d]) == 0:
            continue
        inv_dir = (-e for e in d)
        src = get_slice_before_ghost_layer(inv_dir, ghost_layers, thickness=thickness, full_slice=False)
        dst = get_ghost_region_slice(d, ghost_layers, thickness=thickness, full_slice=False)
        src_dst_slice_tuples.append((src, dst))
    return src_dst_slice_tuples


def get_periodic_boundary_functor(stencil, ghost_layers=1, thickness=None):
    """
    Returns a function that applies periodic boundary conditions
    :param stencil: sequence of directions e.g. ( [0,1], [0,-1] ) for y periodicity
    :param ghost_layers: how many ghost layers the array has
    :param thickness: how many of the ghost layers to copy, None means 'all'
    :return: function that takes a single array and applies the periodic copy operation
    """
    src_dst_slice_tuples = get_periodic_boundary_src_dst_slices(stencil, ghost_layers, thickness)

    def functor(pdfs, **_):
        for src_slice, dst_slice in src_dst_slice_tuples:
            pdfs[dst_slice] = pdfs[src_slice]

    return functor


def slice_intersection(slice1, slice2):
    slice1 = [s if not isinstance(s, int) else slice(s, s + 1, None) for s in slice1]
    slice2 = [s if not isinstance(s, int) else slice(s, s + 1, None) for s in slice2]

    new_min = [max(s1.start, s2.start) for s1, s2 in zip(slice1, slice2)]
    new_max = [min(s1.stop, s2.stop) for s1, s2 in zip(slice1, slice2)]
    if any(max_p - min_p < 0 for min_p, max_p in zip(new_min, new_max)):
        return None

    return [slice(min_p, max_p, None) for min_p, max_p in zip(new_min, new_max)]
