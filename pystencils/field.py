import hashlib
import pickle
import re
from enum import Enum
from itertools import chain
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
import sympy as sp
from sympy.core.cache import cacheit

from pystencils.alignedarray import aligned_empty
from pystencils.data_types import StructType, create_type
from pystencils.kernelparameters import FieldShapeSymbol, FieldStrideSymbol
from pystencils.stencil import direction_string_to_offset, offset_to_direction_string
from pystencils.sympyextensions import is_integer_sequence

__all__ = ['Field', 'fields', 'FieldType', 'AbstractField']


def fields(description=None, index_dimensions=0, layout=None, **kwargs):
    """Creates pystencils fields from a string description.

    Examples:
        Create a 2D scalar and vector field:
            >>> s, v = fields("s, v(2): double[2D]")
            >>> assert s.spatial_dimensions == 2 and s.index_dimensions == 0
            >>> assert (v.spatial_dimensions, v.index_dimensions, v.index_shape) == (2, 1, (2,))

        Create an integer field of shape (10, 20):
            >>> f = fields("f : int32[10, 20]")
            >>> f.has_fixed_shape, f.shape
            (True, (10, 20))

        Numpy arrays can be used as template for shape and data type of field:
            >>> arr_s, arr_v = np.zeros([20, 20]), np.zeros([20, 20, 2])
            >>> s, v = fields("s, v(2)", s=arr_s, v=arr_v)
            >>> assert s.index_dimensions == 0 and s.dtype.numpy_dtype == arr_s.dtype
            >>> assert v.index_shape == (2,)


        Format string can be left out, field names are taken from keyword arguments.
            >>> fields(f1=arr_s, f2=arr_s)
            [f1, f2]

        The keyword names ``index_dimension`` and ``layout`` have special meaning, don't use them for field names
            >>> f = fields(f=arr_v, index_dimensions=1)
            >>> assert f.index_dimensions == 1
            >>> f = fields("pdfs(19) : float32[3D]", layout='fzyx')
            >>> f.layout
            (2, 1, 0)
    """
    result = []
    if description:
        field_descriptions, dtype, shape = _parse_description(description)
        layout = 'numpy' if layout is None else layout
        for field_name, idx_shape in field_descriptions:
            if field_name in kwargs:
                arr = kwargs[field_name]
                idx_shape_of_arr = () if not len(idx_shape) else arr.shape[-len(idx_shape):]
                assert idx_shape_of_arr == idx_shape
                f = Field.create_from_numpy_array(field_name, kwargs[field_name], index_dimensions=len(idx_shape))
            elif isinstance(shape, tuple):
                f = Field.create_fixed_size(field_name, shape + idx_shape, dtype=dtype,
                                            index_dimensions=len(idx_shape), layout=layout)
            elif isinstance(shape, int):
                f = Field.create_generic(field_name, spatial_dimensions=shape, dtype=dtype,
                                         index_shape=idx_shape, layout=layout)
            elif shape is None:
                f = Field.create_generic(field_name, spatial_dimensions=2, dtype=dtype,
                                         index_shape=idx_shape, layout=layout)
            else:
                assert False
            result.append(f)
    else:
        assert layout is None, "Layout can not be specified when creating Field from numpy array"
        for field_name, arr in kwargs.items():
            result.append(Field.create_from_numpy_array(field_name, arr, index_dimensions=index_dimensions))

    if len(result) == 0:
        return None
    elif len(result) == 1:
        return result[0]
    else:
        return result


class FieldType(Enum):
    # generic fields
    GENERIC = 0
    # index fields are currently only used for boundary handling
    # the coordinates are not the loop counters in that case, but are read from this index field
    INDEXED = 1
    # communication buffer, used for (un)packing data in communication.
    BUFFER = 2
    # unsafe fields may be accessed in an absolute fashion - the index depends on the data
    # and thus may lead to out-of-bounds accesses
    CUSTOM = 3

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

    @staticmethod
    def is_custom(field):
        assert isinstance(field, Field)
        return field.field_type == FieldType.CUSTOM


class AbstractField:

    class AbstractAccess:
        pass


class Field(AbstractField):
    """
    With fields one can formulate stencil-like update rules on structured grids.
    This Field class knows about the dimension, memory layout (strides) and optionally about the size of an array.

    Creating Fields:
        The preferred method to create fields is the `fields` function.
        Alternatively one can use one of the static functions `Field.create_generic`, `Field.create_from_numpy_array`
        and `Field.create_fixed_size`. Don't instantiate the Field directly!
        Fields can be created with known or unknown shapes:

        1. If you want to create a kernel with fixed loop sizes i.e. the shape of the array is already known.
           This is usually the case if just-in-time compilation directly from Python is done.
           (see `Field.create_from_numpy_array`
        2. create a more general kernel that works for variable array sizes. This can be used to create kernels
           beforehand for a library. (see `Field.create_generic`)

    Dimensions and Indexing:
        A field has spatial and index dimensions, where the spatial dimensions come first.
        The interpretation is that the field has multiple cells in (usually) two or three dimensional space which are
        looped over. Additionally N values are stored per cell. In this case spatial_dimensions is two or three,
        and index_dimensions equals N. If you want to store a matrix on each point in a two dimensional grid, there
        are four dimensions, two spatial and two index dimensions: ``len(arr.shape) == spatial_dims + index_dims``

        The shape of the index dimension does not have to be specified. Just use the 'index_dimensions' parameter.
        However, it is good practice to define the shape, since out of bounds accesses can be directly detected in this
        case. The shape can be passed with the 'index_shape' parameter of the field creation functions.

        When accessing (indexing) a field the result is a `Field.Access` which is derived from sympy Symbol.
        First specify the spatial offsets in [], then in case index_dimension>0 the indices in ()
        e.g. ``f[-1,0,0](7)``

    Example using no index dimensions:
        >>> a = np.zeros([10, 10])
        >>> f = Field.create_from_numpy_array("f", a, index_dimensions=0)
        >>> jacobi = (f[-1,0] + f[1,0] + f[0,-1] + f[0,1]) / 4

    Examples for index dimensions to create LB field and implement stream pull:
        >>> from pystencils import Assignment
        >>> stencil = np.array([[0,0], [0,1], [0,-1]])
        >>> src, dst = fields("src(3), dst(3) : double[2D]")
        >>> assignments = [Assignment(dst[0,0](i), src[-offset](i)) for i, offset in enumerate(stencil)];
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
        if index_shape is not None:
            assert index_dimensions == 0 or index_dimensions == len(index_shape)
            index_dimensions = len(index_shape)
        if isinstance(layout, str):
            layout = spatial_layout_string_to_tuple(layout, dim=spatial_dimensions)

        total_dimensions = spatial_dimensions + index_dimensions
        if index_shape is None or len(index_shape) == 0:
            shape = tuple([FieldShapeSymbol([field_name], i) for i in range(total_dimensions)])
        else:
            shape = tuple([FieldShapeSymbol([field_name], i) for i in range(spatial_dimensions)] + list(index_shape))

        strides = tuple([FieldStrideSymbol(field_name, i) for i in range(total_dimensions)])

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
                          dtype=np.float64, layout: str = 'numpy', strides: Optional[Sequence[int]] = None) -> 'Field':
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
        self._field_name = field_name
        assert isinstance(field_type, FieldType)
        assert len(shape) == len(strides)
        self.field_type = field_type
        self._dtype = create_type(dtype)
        self._layout = normalize_layout(layout)
        self.shape = shape
        self.strides = strides
        self.latex_name = None  # type: Optional[str]

    def new_field_with_different_name(self, new_name):
        if self.has_fixed_shape:
            return Field(new_name, self.field_type, self._dtype, self._layout, self.shape, self.strides)
        else:
            return Field.create_generic(new_name, self.spatial_dimensions, self.dtype.numpy_dtype,
                                        self.index_dimensions, self._layout, self.index_shape, self.field_type)

    @property
    def spatial_dimensions(self) -> int:
        return len(self._layout)

    @property
    def index_dimensions(self) -> int:
        return len(self.shape) - len(self._layout)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def layout(self):
        return self._layout

    @property
    def name(self) -> str:
        return self._field_name

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
        return self._field_name

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
            return sp.Matrix([self.center])
        if len(index_shape) == 1:
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

    def absolute_access(self, offset, index):
        assert FieldType.is_custom(self)
        return Field.Access(self, offset, index, is_absolute_access=True)

    def __call__(self, *args, **kwargs):
        center = tuple([0] * self.spatial_dimensions)
        return Field.Access(self, center)(*args, **kwargs)

    def hashable_contents(self):
        dth = hash(self._dtype)
        return self._layout, self.shape, self.strides, dth, self.field_type, self._field_name, self.latex_name

    def __hash__(self):
        return hash(self.hashable_contents())

    def __eq__(self, other):
        if not isinstance(other, Field):
            return False
        return self.hashable_contents() == other.hashable_contents()

    # noinspection PyAttributeOutsideInit,PyUnresolvedReferences
    class Access(sp.Symbol, AbstractField.AbstractAccess):
        """Class representing a relative access into a `Field`.

        This class behaves like a normal sympy Symbol, it is actually derived from it. One can built up
        sympy expressions using field accesses, solve for them, etc.

        Examples:
            >>> vector_field_2d = fields("v(2): double[2D]")  # create a 2D vector field
            >>> northern_neighbor_y_component = vector_field_2d[0, 1](1)
            >>> northern_neighbor_y_component
            v_N^1
            >>> central_y_component = vector_field_2d(1)
            >>> central_y_component
            v_C^1
            >>> central_y_component.get_shifted(1, 0)  # move the existing access
            v_E^1
            >>> central_y_component.at_index(0)  # change component
            v_C^0
        """
        def __new__(cls, name, *args, **kwargs):
            obj = Field.Access.__xnew_cached_(cls, name, *args, **kwargs)
            return obj

        def __new_stage2__(self, field, offsets=(0, 0, 0), idx=None, is_absolute_access=False):
            field_name = field.name
            offsets_and_index = (*offsets, *idx) if idx is not None else offsets
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
                offset_name = hashlib.md5(pickle.dumps(offsets_and_index)).hexdigest()[:12]
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
            obj._offsets = tuple(obj._offsets)
            obj._offsetName = offset_name
            obj._superscript = superscript
            obj._index = idx

            obj._indirect_addressing_fields = set()
            for e in chain(obj._offsets, obj._index):
                if isinstance(e, sp.Basic):
                    obj._indirect_addressing_fields.update(a.field for a in e.atoms(Field.Access))

            obj._is_absolute_access = is_absolute_access
            return obj

        def __getnewargs__(self):
            return self.field, self.offsets, self.index, self.is_absolute_access

        # noinspection SpellCheckingInspection
        __xnew__ = staticmethod(__new_stage2__)
        # noinspection SpellCheckingInspection
        __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

        def __call__(self, *idx):
            if self._index != tuple([0] * self.field.index_dimensions):
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
        def field(self) -> 'Field':
            """Field that the Access points to"""
            return self._field

        @property
        def offsets(self) -> Tuple:
            """Spatial offset as tuple"""
            return self._offsets

        @property
        def required_ghost_layers(self) -> int:
            """Largest spatial distance that is accessed."""
            return int(np.max(np.abs(self._offsets)))

        @property
        def nr_of_coordinates(self):
            return len(self._offsets)

        @property
        def offset_name(self) -> str:
            """Spatial offset as string, East-West for x, North-South for y and Top-Bottom for z coordinate.

            Example:
                >>> f = fields("f: double[2D]")
                >>> f[1, 1].offset_name  # north-east
                'NE'
            """
            return self._offsetName

        @property
        def index(self):
            """Value of index coordinates as tuple."""
            return self._index

        def neighbor(self, coord_id: int, offset: int) -> 'Field.Access':
            """Returns a new Access with changed spatial coordinates.

            Args:
                coord_id: index of the coordinate to change (0 for x, 1 for y,...)
                offset: incremental change of this coordinate

            Example:
                >>> f = fields('f: [2D]')
                >>> f[0,0].neighbor(coord_id=1, offset=-1)
                f_S
            """
            offset_list = list(self.offsets)
            offset_list[coord_id] += offset
            return Field.Access(self.field, tuple(offset_list), self.index)

        def get_shifted(self, *shift) -> 'Field.Access':
            """Returns a new Access with changed spatial coordinates

            Example:
                >>> f = fields("f: [2D]")
                >>> f[0,0].get_shifted(1, 1)
                f_NE
            """
            return Field.Access(self.field, tuple(a + b for a, b in zip(shift, self.offsets)), self.index)

        def at_index(self, *idx_tuple) -> 'Field.Access':
            """Returns new Access with changed index.

            Example:
                >>> f = fields("f(9): [2D]")
                >>> f(0).at_index(8)
                f_C^8
            """
            return Field.Access(self.field, self.offsets, idx_tuple)

        @property
        def is_absolute_access(self) -> bool:
            """Indicates if a field access is relative to the loop counters (this is the default) or absolute"""
            return self._is_absolute_access

        @property
        def indirect_addressing_fields(self) -> Set['Field']:
            """Returns a set of fields that the access depends on.

             e.g. f[index_field[1, 0]], the outer access to f depends on index_field
             """
            return self._indirect_addressing_fields

        def _hashable_content(self):
            super_class_contents = super(Field.Access, self)._hashable_content()
            return (super_class_contents, self._field.hashable_contents(), *self._index, *self._offsets)

        def _latex(self, _):
            n = self._field.latex_name if self._field.latex_name else self._field.name
            offset_str = ",".join([sp.latex(o) for o in self.offsets])
            if self.is_absolute_access:
                offset_str = "\\mathbf{}".format(offset_str)
            elif self.field.spatial_dimensions > 1:
                offset_str = "({})".format(offset_str)

            if self.index and self.index != (0,):
                return "{{%s}_{%s}^{%s}}" % (n, offset_str, self.index if len(self.index) > 1 else self.index[0])
            else:
                return "{{%s}_{%s}}" % (n, offset_str)

        def __str__(self):
            n = self._field.latex_name if self._field.latex_name else self._field.name
            offset_str = ",".join([sp.latex(o) for o in self.offsets])
            if self.is_absolute_access:
                offset_str = "[abs]{}".format(offset_str)
            if self.index and self.index != (0,):
                return "%s[%s](%s)" % (n, offset_str, self.index if len(self.index) > 1 else self.index[0])
            else:
                return "%s[%s]" % (n, offset_str)


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
        return tuple(reversed(range(dim - 1))) + (dim - 1,)
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


# ---------------------------------------- Parsing of string in fields() function --------------------------------------

field_description_regex = re.compile(r"""
    \s*                 # ignore leading white spaces
    (\w+)               # identifier is a sequence of alphanumeric characters, is stored in first group
    (?:                 # optional index specification e.g. (1, 4, 2)
        \s*
        \(
            ([^\)]+)    # read everything up to closing bracket
        \)
        \s*
    )?
    \s*,?\s*             # ignore trailing white spaces and comma
""", re.VERBOSE)

type_description_regex = re.compile(r"""
    \s*
    (\w+)?       # optional dtype
    \s*
    \[
        ([^\]]+)
    \]
    \s*
""", re.VERBOSE | re.IGNORECASE)


def _parse_description(description):
    def parse_part1(d):
        result = field_description_regex.match(d)
        while result:
            name, index_str = result.group(1), result.group(2)
            index = tuple(int(e) for e in index_str.split(",")) if index_str else ()
            yield name, index
            d = d[result.end():]
            result = field_description_regex.match(d)

    def parse_part2(d):
        result = type_description_regex.match(d)
        if result:
            data_type_str, size_info = result.group(1), result.group(2).strip().lower()
            if data_type_str is None:
                data_type_str = 'float64'
            data_type_str = data_type_str.lower().strip()

            if not data_type_str:
                data_type_str = 'float64'
            if size_info.endswith('d'):
                size_info = int(size_info[:-1])
            else:
                size_info = tuple(int(e) for e in size_info.split(","))
            return data_type_str, size_info
        else:
            raise ValueError("Could not parse field description")

    if ':' in description:
        field_description, field_info = description.split(':')
    else:
        field_description, field_info = description, 'float64[2D]'

    fields_info = [e for e in parse_part1(field_description)]
    if not field_info:
        raise ValueError("Could not parse field description")

    data_type, size = parse_part2(field_info)
    return fields_info, data_type, size
