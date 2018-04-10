from enum import Enum
from itertools import chain
from typing import Tuple, Sequence, Optional, List
import numpy as np
import sympy as sp
from sympy.core.cache import cacheit
from sympy.tensor import IndexedBase

from pystencils.assignment import Assignment
from pystencils.alignedarray import aligned_empty
from pystencils.data_types import TypedSymbol, create_type, create_composite_type_from_string, StructType
from pystencils.sympyextensions import is_integer_sequence


class FieldType(Enum):
    # generic fields
    GENERIC = 0
    # index fields are currently only used for boundary handling
    # the coordinates are not the loop counters in that case, but are read from this index field
    INDEXED = 1
    # communication buffer, used for (un)packing data in communication.
    BUFFER = 2

    @staticmethod
    def is_generic(field):
        assert isinstance(field, Field)
        return field.field_type == FieldType.GENERIC

    @staticmethod
    def is_indexed(field):
        assert isinstance(field, Field)
        return field.field_type == FieldType.INDEXED

    @staticmethod
    def is_buffer(field):
        assert isinstance(field, Field)
        return field.field_type == FieldType.BUFFER


class Field(object):
    """
    With fields one can formulate stencil-like update rules on structured grids.
    This Field class knows about the dimension, memory layout (strides) and optionally about the size of an array.

    Creating Fields:

        To create a field use one of the static create* members. There are two options:

        1. create a kernel with fixed loop sizes i.e. the shape of the array is already known. This is usually the
           case if just-in-time compilation directly from Python is done. (see :func:`Field.create_from_numpy_array`)
        2. create a more general kernel that works for variable array sizes. This can be used to create kernels
           beforehand for a library. (see :func:`Field.create_generic`)

    Dimensions:
        A field has spatial and index dimensions, where the spatial dimensions come first.
        The interpretation is that the field has multiple cells in (usually) two or three dimensional space which are
        looped over. Additionally  N values are stored per cell. In this case spatial_dimensions is two or three,
        and index_dimensions equals N. If you want to store a matrix on each point in a two dimensional grid, there
        are four dimensions, two spatial and two index dimensions: ``len(arr.shape) == spatial_dims + index_dims``

    Indexing:
        When accessing (indexing) a field the result is a FieldAccess which is derived from sympy Symbol.
        First specify the spatial offsets in [], then in case index_dimension>0 the indices in ()
        e.g. ``f[-1,0,0](7)``

    Example without index dimensions:
        >>> a = np.zeros([10, 10])
        >>> f = Field.create_from_numpy_array("f", a, index_dimensions=0)
        >>> jacobi = ( f[-1,0] + f[1,0] + f[0,-1] + f[0,1] ) / 4

    Example with index dimensions: LBM D2Q9 stream pull
        >>> stencil = np.array([[0,0], [0,1], [0,-1]])
        >>> src = Field.create_generic("src", spatial_dimensions=2, index_dimensions=1)
        >>> dst = Field.create_generic("dst", spatial_dimensions=2, index_dimensions=1)
        >>> for i, offset in enumerate(stencil):
        ...     Assignment(dst[0,0](i), src[-offset](i))
        Assignment(dst_C^0, src_C^0)
        Assignment(dst_C^1, src_S^1)
        Assignment(dst_C^2, src_N^2)
    """

    @staticmethod
    def create_generic(field_name, spatial_dimensions, dtype=np.float64, index_dimensions=0, layout='numpy',
                       index_shape=None, field_type=FieldType.GENERIC) -> 'Field':
        """
        Creates a generic field where the field size is not fixed i.e. can be called with arrays of different sizes

        Args:
            field_name: symbolic name for the field
            dtype: numpy data type of the array the kernel is called with later
            spatial_dimensions: see documentation of Field
            index_dimensions: see documentation of Field
            layout: tuple specifying the loop ordering of the spatial dimensions e.g. (2, 1, 0 ) means that
                    the outer loop loops over dimension 2, the second outer over dimension 1, and the inner loop
                    over dimension 0. Also allowed: the strings 'numpy' (0,1,..d) or 'reverse_numpy' (d, ..., 1, 0)
            index_shape: optional shape of the index dimensions i.e. maximum values allowed for each index dimension,
                        has to be a list or tuple
            field_type: besides the normal GENERIC fields, there are INDEXED fields that store indices of the domain
                        that should be iterated over, and BUFFER fields that are used to generate
                        communication packing/unpacking kernels
        """
        if isinstance(layout, str):
            layout = spatial_layout_string_to_tuple(layout, dim=spatial_dimensions)
        shape_symbol = IndexedBase(TypedSymbol(Field.SHAPE_PREFIX + field_name, Field.SHAPE_DTYPE), shape=(1,))
        stride_symbol = IndexedBase(TypedSymbol(Field.STRIDE_PREFIX + field_name, Field.STRIDE_DTYPE), shape=(1,))
        total_dimensions = spatial_dimensions + index_dimensions
        if index_shape is None or len(index_shape) == 0:
            shape = tuple([shape_symbol[i] for i in range(total_dimensions)])
        else:
            shape = tuple([shape_symbol[i] for i in range(spatial_dimensions)] + list(index_shape))

        strides = tuple([stride_symbol[i] for i in range(total_dimensions)])

        np_data_type = np.dtype(dtype)
        if np_data_type.fields is not None:
            if index_dimensions != 0:
                raise ValueError("Structured arrays/fields are not allowed to have an index dimension")
            shape += (1,)
            strides += (1,)

        return Field(field_name, field_type, dtype, layout, shape, strides)

    @staticmethod
    def create_from_numpy_array(field_name: str, array: np.ndarray, index_dimensions: int = 0) -> 'Field':
        """Creates a field based on the layout, data type, and shape of a given numpy array.

        Kernels created for these kind of fields can only be called with arrays of the same layout, shape and type.

        Args:
            field_name: symbolic name for the field
            array: numpy array
            index_dimensions: see documentation of Field
        """
        spatial_dimensions = len(array.shape) - index_dimensions
        if spatial_dimensions < 1:
            raise ValueError("Too many index dimensions. At least one spatial dimension required")

        full_layout = get_layout_of_array(array)
        spatial_layout = tuple([i for i in full_layout if i < spatial_dimensions])
        assert len(spatial_layout) == spatial_dimensions

        strides = tuple([s // np.dtype(array.dtype).itemsize for s in array.strides])
        shape = tuple(int(s) for s in array.shape)

        numpy_dtype = np.dtype(array.dtype)
        if numpy_dtype.fields is not None:
            if index_dimensions != 0:
                raise ValueError("Structured arrays/fields are not allowed to have an index dimension")
            shape += (1,)
            strides += (1,)

        return Field(field_name, FieldType.GENERIC, array.dtype, spatial_layout, shape, strides)

    @staticmethod
    def create_fixed_size(field_name: str, shape: Tuple[int, ...], index_dimensions: int = 0,
                          dtype=np.float64, layout: str = 'numpy', strides: Optional[Sequence[int]]=None) -> 'Field':
        """
        Creates a field with fixed sizes i.e. can be called only with arrays of the same size and layout

        Args:
            field_name: symbolic name for the field
            shape: overall shape of the array
            index_dimensions: how many of the trailing dimensions are interpreted as index (as opposed to spatial)
            dtype: numpy data type of the array the kernel is called with later
            layout: full layout of array, not only spatial dimensions
            strides: strides in bytes or None to automatically compute them from shape (assuming no padding)
        """
        spatial_dimensions = len(shape) - index_dimensions
        assert spatial_dimensions >= 1

        if isinstance(layout, str):
            layout = layout_string_to_tuple(layout, spatial_dimensions + index_dimensions)

        shape = tuple(int(s) for s in shape)
        if strides is None:
            strides = compute_strides(shape, layout)
        else:
            assert len(strides) == len(shape)
            strides = tuple([s // np.dtype(dtype).itemsize for s in strides])

        numpy_dtype = np.dtype(dtype)
        if numpy_dtype.fields is not None:
            if index_dimensions != 0:
                raise ValueError("Structured arrays/fields are not allowed to have an index dimension")
            shape += (1,)
            strides += (1,)

        spatial_layout = list(layout)
        for i in range(spatial_dimensions, len(layout)):
            spatial_layout.remove(i)
        return Field(field_name, FieldType.GENERIC, dtype, tuple(spatial_layout), shape, strides)

    def __init__(self, field_name, field_type, dtype, layout, shape, strides):
        """Do not use directly. Use static create* methods"""
        self._fieldName = field_name
        assert isinstance(field_type, FieldType)
        self.field_type = field_type
        self._dtype = create_type(dtype)
        self._layout = normalize_layout(layout)
        self.shape = shape
        self.strides = strides
        self.latex_name: Optional[str] = None

    def new_field_with_different_name(self, new_name):
        return Field(new_name, self.field_type, self._dtype, self._layout, self.shape, self.strides)

    @property
    def spatial_dimensions(self) -> int:
        return len(self._layout)

    @property
    def index_dimensions(self) -> int:
        return len(self.shape) - len(self._layout)

    @property
    def layout(self):
        return self._layout

    @property
    def name(self) -> str:
        return self._fieldName

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        return self.shape[:self.spatial_dimensions]

    @property
    def has_fixed_shape(self):
        return is_integer_sequence(self.shape)

    @property
    def index_shape(self):
        return self.shape[self.spatial_dimensions:]

    @property
    def has_fixed_index_shape(self):
        return is_integer_sequence(self.index_shape)

    @property
    def spatial_strides(self):
        return self.strides[:self.spatial_dimensions]

    @property
    def index_strides(self):
        return self.strides[self.spatial_dimensions:]

    @property
    def dtype(self):
        return self._dtype

    def __repr__(self):
        return self._fieldName

    def neighbor(self, coord_id, offset):
        offset_list = [0] * self.spatial_dimensions
        offset_list[coord_id] = offset
        return Field.Access(self, tuple(offset_list))

    def neighbors(self, stencil):
        return [self.__getitem__(s) for s in stencil]

    @property
    def center_vector(self):
        index_shape = self.index_shape
        if len(index_shape) == 0:
            return self.center
        elif len(index_shape) == 1:
            return sp.Matrix([self(i) for i in range(index_shape[0])])
        elif len(index_shape) == 2:
            def cb(*args):
                r = self.__call__(*args)
                return r
            return sp.Matrix(*index_shape, cb)

    @property
    def center(self):
        center = tuple([0] * self.spatial_dimensions)
        return Field.Access(self, center)

    def __getitem__(self, offset):
        if type(offset) is np.ndarray:
            offset = tuple(offset)
        if type(offset) is str:
            offset = tuple(direction_string_to_offset(offset, self.spatial_dimensions))
        if type(offset) is not tuple:
            offset = (offset,)
        if len(offset) != self.spatial_dimensions:
            raise ValueError("Wrong number of spatial indices: "
                             "Got %d, expected %d" % (len(offset), self.spatial_dimensions))
        return Field.Access(self, offset)

    def __call__(self, *args, **kwargs):
        center = tuple([0] * self.spatial_dimensions)
        return Field.Access(self, center)(*args, **kwargs)

    def __hash__(self):
        return hash((self._layout, self.shape, self.strides, self._dtype, self.field_type, self._fieldName))

    def __eq__(self, other):
        self_tuple = (self.shape, self.strides, self.name, self.dtype, self.field_type)
        other_tuple = (other.shape, other.strides, other.name, other.dtype, other.field_type)
        return self_tuple == other_tuple

    PREFIX = "f"
    STRIDE_PREFIX = PREFIX + "stride_"
    SHAPE_PREFIX = PREFIX + "shape_"
    STRIDE_DTYPE = create_composite_type_from_string("const int *")
    SHAPE_DTYPE = create_composite_type_from_string("const int *")
    DATA_PREFIX = PREFIX + "d_"

    # noinspection PyAttributeOutsideInit,PyUnresolvedReferences
    class Access(sp.Symbol):
        def __new__(cls, name, *args, **kwargs):
            obj = Field.Access.__xnew_cached_(cls, name, *args, **kwargs)
            return obj

        def __new_stage2__(self, field, offsets=(0, 0, 0), idx=None):
            field_name = field.name
            offsets_and_index = chain(offsets, idx) if idx is not None else offsets
            constant_offsets = not any([isinstance(o, sp.Basic) and not o.is_Integer for o in offsets_and_index])

            if not idx:
                idx = tuple([0] * field.index_dimensions)

            if constant_offsets:
                offset_name = offset_to_direction_string(offsets)
                if field.index_dimensions == 0:
                    superscript = None
                elif field.index_dimensions == 1:
                    superscript = str(idx[0])
                else:
                    idx_str = ",".join([str(e) for e in idx])
                    superscript = idx_str
                if field.has_fixed_index_shape and not isinstance(field.dtype, StructType):
                    for i, bound in zip(idx, field.index_shape):
                        if i >= bound:
                            raise ValueError("Field index out of bounds")
            else:
                offset_name = "%0.10X" % (abs(hash(tuple(offsets_and_index))))
                superscript = None

            symbol_name = "%s_%s" % (field_name, offset_name)
            if superscript is not None:
                symbol_name += "^" + superscript

            obj = super(Field.Access, self).__xnew__(self, symbol_name)
            obj._field = field
            obj._offsets = []
            for o in offsets:
                if isinstance(o, sp.Basic):
                    obj._offsets.append(o)
                else:
                    obj._offsets.append(int(o))
            obj._offsetName = offset_name
            obj._superscript = superscript
            obj._index = idx

            return obj

        def __getnewargs__(self):
            return self.field, self.offsets, self.index

        # noinspection SpellCheckingInspection
        __xnew__ = staticmethod(__new_stage2__)
        # noinspection SpellCheckingInspection
        __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

        def __call__(self, *idx):
            if self._index != tuple([0]*self.field.index_dimensions):
                raise ValueError("Indexing an already indexed Field.Access")

            idx = tuple(idx)

            if self.field.index_dimensions == 0 and idx == (0,):
                idx = ()

            if len(idx) != self.field.index_dimensions:
                raise ValueError("Wrong number of indices: "
                                 "Got %d, expected %d" % (len(idx), self.field.index_dimensions))
            return Field.Access(self.field, self._offsets, idx)

        def __getitem__(self, *idx):
            return self.__call__(*idx)

        def __iter__(self):
            """This is necessary to work with parts of sympy that test if an object is iterable (e.g. simplify).
            The __getitem__ would make it iterable"""
            raise TypeError("Field access is not iterable")

        @property
        def field(self):
            return self._field

        @property
        def offsets(self):
            return self._offsets

        @offsets.setter
        def offsets(self, value):
            self._offsets = value

        @property
        def required_ghost_layers(self):
            return int(np.max(np.abs(self._offsets)))

        @property
        def nr_of_coordinates(self):
            return len(self._offsets)

        @property
        def offset_name(self) -> str:
            return self._offsetName

        @property
        def index(self):
            return self._index

        def get_neighbor(self, *offsets) -> 'Field.Access':
            return Field.Access(self.field, offsets, self.index)

        def neighbor(self, coord_id: int, offset: Sequence[int]) -> 'Field.Access':
            offset_list = list(self.offsets)
            offset_list[coord_id] += offset
            return Field.Access(self.field, tuple(offset_list), self.index)

        def get_shifted(self, *shift)-> 'Field.Access':
            return Field.Access(self.field, tuple(a + b for a, b in zip(shift, self.offsets)), self.index)

        def _hashable_content(self):
            super_class_contents = list(super(Field.Access, self)._hashable_content())
            t = tuple(super_class_contents + [hash(self._field), self._index] + self._offsets)
            return t

        def _latex(self, _):
            n = self._field.latex_name if self._field.latex_name else self._field.name
            if self._superscript:
                return "{{%s}_{%s}^{%s}}" % (n, self._offsetName, self._superscript)
            else:
                return "{{%s}_{%s}}" % (n, self._offsetName)


def get_layout_from_strides(strides: Sequence[int], index_dimension_ids: Optional[List[int]] = None):
    index_dimension_ids = [] if index_dimension_ids is None else index_dimension_ids
    coordinates = list(range(len(strides)))
    relevant_strides = [stride for i, stride in enumerate(strides) if i not in index_dimension_ids]
    result = [x for (y, x) in sorted(zip(relevant_strides, coordinates), key=lambda pair: pair[0], reverse=True)]
    return normalize_layout(result)


def get_layout_of_array(arr: np.ndarray, index_dimension_ids: Optional[List[int]] = None):
    """ Returns a list indicating the memory layout (linearization order) of the numpy array.

    Examples:
        >>> get_layout_of_array(np.zeros([3,3,3]))
        (0, 1, 2)

    In this example the loop over the zeroth coordinate should be the outermost loop,
    followed by the first and second. Elements arr[x,y,0] and arr[x,y,1] are adjacent in memory.
    Normally constructed numpy arrays have this order, however by stride tricks or other frameworks, arrays
    with different memory layout can be created.

    The index_dimension_ids parameter leaves specifies which coordinates should not be
    """
    index_dimension_ids = [] if index_dimension_ids is None else index_dimension_ids
    return get_layout_from_strides(arr.strides, index_dimension_ids)


def create_numpy_array_with_layout(shape, layout, alignment=False, byte_offset=0, **kwargs):
    """Creates numpy array with given memory layout.

    Args:
        shape: shape of the resulting array
        layout: layout as tuple, where the coordinates are ordered from slow to fast
        alignment: number of bytes to align the beginning and the innermost coordinate to, or False for no alignment
        byte_offset: only used when alignment is specified, align not beginning but address at this offset
                     mostly used to align first inner cell, not ghost cells

    Example:
        >>> res = create_numpy_array_with_layout(shape=(2, 3, 4, 5), layout=(3, 2, 0, 1))
        >>> res.shape
        (2, 3, 4, 5)
        >>> get_layout_of_array(res)
        (3, 2, 0, 1)
    """
    assert set(layout) == set(range(len(shape))), "Wrong layout descriptor"
    cur_layout = list(range(len(shape)))
    swaps = []
    for i in range(len(layout)):
        if cur_layout[i] != layout[i]:
            index_to_swap_with = cur_layout.index(layout[i])
            swaps.append((i, index_to_swap_with))
            cur_layout[i], cur_layout[index_to_swap_with] = cur_layout[index_to_swap_with], cur_layout[i]
    assert tuple(cur_layout) == tuple(layout)

    shape = list(shape)
    for a, b in swaps:
        shape[a], shape[b] = shape[b], shape[a]

    if not alignment:
        res = np.empty(shape, order='c', **kwargs)
    else:
        if alignment is True:
            alignment = 8 * 4
        res = aligned_empty(shape, alignment, byte_offset=byte_offset, **kwargs)

    for a, b in reversed(swaps):
        res = res.swapaxes(a, b)
    return res


def spatial_layout_string_to_tuple(layout_str: str, dim: int) -> Tuple[int, ...]:
    if layout_str in ('fzyx', 'zyxf'):
        assert dim <= 3
        return tuple(reversed(range(dim)))

    if layout_str in ('fzyx', 'f', 'reverse_numpy', 'SoA'):
        return tuple(reversed(range(dim)))
    elif layout_str in ('c', 'numpy', 'AoS'):
        return tuple(range(dim))
    raise ValueError("Unknown layout descriptor " + layout_str)


def layout_string_to_tuple(layout_str, dim):
    layout_str = layout_str.lower()
    if layout_str == 'fzyx' or layout_str == 'soa':
        assert dim <= 4
        return tuple(reversed(range(dim)))
    elif layout_str == 'zyxf' or layout_str == 'aos':
        assert dim <= 4
        return tuple(reversed(range(dim - 1))) + (dim-1,)
    elif layout_str == 'f' or layout_str == 'reverse_numpy':
        return tuple(reversed(range(dim)))
    elif layout_str == 'c' or layout_str == 'numpy':
        return tuple(range(dim))
    raise ValueError("Unknown layout descriptor " + layout_str)


def normalize_layout(layout):
    """Takes a layout tuple and subtracts the minimum from all entries"""
    min_entry = min(layout)
    return tuple(i - min_entry for i in layout)


def compute_strides(shape, layout):
    """
    Computes strides assuming no padding exists

    Args:
        shape: shape (size) of array
        layout: layout specification as tuple

    Returns:
        strides in elements, not in bytes
    """
    dim = len(shape)
    assert len(layout) == dim
    assert len(set(layout)) == dim
    strides = [0] * dim
    product = 1
    for j in reversed(layout):
        strides[j] = product
        product *= shape[j]
    return tuple(strides)


def offset_component_to_direction_string(coordinate_id: int, value: int) -> str:
    """Translates numerical offset to string notation.

    x offsets are labeled with east 'E' and 'W',
    y offsets with north 'N' and 'S' and
    z offsets with top 'T' and bottom 'B'
    If the absolute value of the offset is bigger than 1, this number is prefixed.

    Args:
        coordinate_id: integer 0, 1 or 2 standing for x,y and z
        value: integer offset

    Examples:
        >>> offset_component_to_direction_string(0, 1)
        'E'
        >>> offset_component_to_direction_string(1, 2)
        '2N'
    """
    name_components = (('W', 'E'),  # west, east
                       ('S', 'N'),  # south, north
                       ('B', 'T'),  # bottom, top
                       )
    if value == 0:
        result = ""
    elif value < 0:
        result = name_components[coordinate_id][0]
    else:
        result = name_components[coordinate_id][1]
    if abs(value) > 1:
        result = "%d%s" % (abs(value), result)
    return result


def offset_to_direction_string(offsets: Sequence[int]) -> str:
    """
    Translates numerical offset to string notation.
    For details see :func:`offset_component_to_direction_string`
    Args:
        offsets: 3-tuple with x,y,z offset

    Examples:
        >>> offset_to_direction_string([1, -1, 0])
        'SE'
        >>> offset_to_direction_string(([-3, 0, -2]))
        '2B3W'
    """
    names = ["", "", ""]
    for i in range(len(offsets)):
        names[i] = offset_component_to_direction_string(i, offsets[i])
    name = "".join(reversed(names))
    if name == "":
        name = "C"
    return name


def direction_string_to_offset(direction: str, dim: int = 3):
    """
    Reverse mapping of :func:`offset_to_direction_string`

    Args:
        direction: string representation of offset
        dim: dimension of offset, i.e the length of the returned list

    Examples:
        >>> direction_string_to_offset('NW', dim=3)
        array([-1,  1,  0])
        >>> direction_string_to_offset('NW', dim=2)
        array([-1,  1])
        >>> direction_string_to_offset(offset_to_direction_string((3,-2,1)))
        array([ 3, -2,  1])
    """
    offset_dict = {
        'C': np.array([0, 0, 0]),

        'W': np.array([-1, 0, 0]),
        'E': np.array([1, 0, 0]),

        'S': np.array([0, -1, 0]),
        'N': np.array([0, 1, 0]),

        'B': np.array([0, 0, -1]),
        'T': np.array([0, 0, 1]),
    }
    offset = np.array([0, 0, 0])

    while len(direction) > 0:
        factor = 1
        first_non_digit = 0
        while direction[first_non_digit].isdigit():
            first_non_digit += 1
        if first_non_digit > 0:
            factor = int(direction[:first_non_digit])
            direction = direction[first_non_digit:]
        cur_offset = offset_dict[direction[0]]
        offset += factor * cur_offset
        direction = direction[1:]
    return offset[:dim]
