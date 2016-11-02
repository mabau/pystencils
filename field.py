from itertools import chain
import numpy as np
import sympy as sp
from sympy.core.cache import cacheit
from sympy.tensor import IndexedBase
from pystencils.typedsymbol import TypedSymbol


def getLayoutFromNumpyArray(arr):
    """
    Returns a list indicating the memory layout (linearization order) of the numpy array.
    Example:
        >>> getLayoutFromNumpyArray(np.zeros([3,3,3]))
        [0, 1, 2]
    In this example the loop over the zeroth coordinate should be the outermost loop,
    followed by the first and second. Elements arr[x,y,0] and arr[x,y,1] are adjacent in memory.
    Normally constructed numpy arrays have this order, however by stride tricks or other frameworks, arrays
    with different memory layout can be created.
    """
    coordinates = list(range(len(arr.shape)))
    return [x for (y, x) in sorted(zip(arr.strides, coordinates), key=lambda pair: pair[0], reverse=True)]


def numpyDataTypeToC(dtype):
    """Mapping numpy data types to C data types"""
    if dtype == np.float64:
        return "double"
    elif dtype == np.float32:
        return "float"
    elif dtype == np.int32:
        return "int"
    raise NotImplementedError()


def offsetComponentToDirectionString(coordinateId, value):
    """
    Translates numerical offset to string notation.
    x offsets are labeled with east 'E' and 'W',
    y offsets with north 'N' and 'S' and
    z offsets with top 'T' and bottom 'B'
    If the absolute value of the offset is bigger than 1, this number is prefixed.
    :param coordinateId: integer 0, 1 or 2 standing for x,y and z
    :param value: integer offset

    Example:
    >>> offsetComponentToDirectionString(0, 1)
    'E'
    >>> offsetComponentToDirectionString(1, 2)
    '2N'
    """
    nameComponents = (('W', 'E'),  # west, east
                      ('S', 'N'),  # south, north
                      ('B', 'T'),  # bottom, top
                      )
    if value == 0:
        result = ""
    elif value < 0:
        result = nameComponents[coordinateId][0]
    else:
        result = nameComponents[coordinateId][1]
    if abs(value) > 1:
        result = "%d%s" % (abs(value), result)
    return result


def offsetToDirectionString(offsetTuple):
    """
    Translates numerical offset to string notation.
    For details see :func:`offsetComponentToDirectionString`
    :param offsetTuple: 3-tuple with x,y,z offset

    Example:
    >>> offsetToDirectionString([1, -1, 0])
    'SE'
    >>> offsetToDirectionString(([-3, 0, -2]))
    '2B3W'
    """
    names = ["", "", ""]
    for i in range(len(offsetTuple)):
        names[i] = offsetComponentToDirectionString(i, offsetTuple[i])
    name = "".join(reversed(names))
    if name == "":
        name = "C"
    return name


def directionStringToOffset(directionStr, dim=3):
    """
    Reverse mapping of :func:`offsetToDirectionString`
    :param directionStr: string representation of offset
    :param dim: dimension of offset, i.e the length of the returned list

    >>> directionStringToOffset('NW', dim=3)
    array([-1,  1,  0])
    >>> directionStringToOffset('NW', dim=2)
    array([-1,  1])
    >>> directionStringToOffset(offsetToDirectionString([3,-2,1]))
    array([ 3, -2,  1])
    """
    offsetMap = {
        'C': np.array([0, 0, 0]),

        'W': np.array([-1, 0, 0]),
        'E': np.array([1, 0, 0]),

        'S': np.array([0, -1, 0]),
        'N': np.array([0, 1, 0]),

        'B': np.array([0, 0, -1]),
        'T': np.array([0, 0, 1]),
    }
    offset = np.array([0, 0, 0])

    while len(directionStr) > 0:
        factor = 1
        firstNonDigit = 0
        while directionStr[firstNonDigit].isdigit():
            firstNonDigit += 1
        if firstNonDigit > 0:
            factor = int(directionStr[:firstNonDigit])
            directionStr = directionStr[firstNonDigit:]
        curOffset = offsetMap[directionStr[0]]
        offset += factor * curOffset
        directionStr = directionStr[1:]
    return offset[:dim]


class Field:
    """
    With fields one can formulate stencil-like update rules on structured grids.
    This Field class knows about the dimension, memory layout (strides) and optionally about the size of an array.

    To create a field use one of the static create* members. There are two options:
        1. create a kernel with fixed loop sizes i.e. the shape of the array is already known. This is usually the
           case if just-in-time compilation directly from Python is done. (see Field.createFromNumpyArray)
        2. create a more general kernel that works for variable array sizes. This can be used to create kernels
           beforehand for a library. (see Field.createGeneric)

    Dimensions:
        A field has spatial and index dimensions, where the spatial dimensions come first.
        The interpretation is that the field has multiple cells in (usually) two or three dimensional space which are
        looped over. Additionally  N values are stored per cell. In this case spatialDimensions is two or three,
        and indexDimensions equals N. If you want to store a matrix on each point in a two dimensional grid, there
        are four dimensions, two spatial and two index dimensions. len(arr.shape) == spatialDims + indexDims

    Indexing:
        When accessing (indexing) a field the result is a FieldAccess which is derived from sympy Symbol.
        First specify the spatial offsets in [], then in case indexDimension>0 the indices in ()
        e.g. f[-1,0,0](7)

    Example without index dimensions:
        >>> a = np.zeros([10, 10])
        >>> f = Field.createFromNumpyArray("f", a, indexDimensions=0)
        >>> jacobi = ( f[-1,0] + f[1,0] + f[0,-1] + f[0,1] ) / 4

    Example with index dimensions: LBM D2Q9 stream pull
        >>> stencil = np.array([[0,0], [0,1], [0,-1]])
        >>> src = Field.createGeneric("src", spatialDimensions=2, indexDimensions=1)
        >>> dst = Field.createGeneric("dst", spatialDimensions=2, indexDimensions=1)
        >>> for i, offset in enumerate(stencil):
        ...     sp.Eq(dst[0,0](i), src[-offset](i))
        Eq(dst_C^0, src_C^0)
        Eq(dst_C^1, src_S^1)
        Eq(dst_C^2, src_N^2)
    """
    @staticmethod
    def createFromNumpyArray(fieldName, npArray, indexDimensions=0):
        """
        Creates a field based on the layout, data type, and shape of a given numpy array.
        Kernels created for these kind of fields can only be called with arrays of the same layout, shape and type.
        :param fieldName: symbolic name for the field
        :param npArray: numpy array
        :param indexDimensions: see documentation of Field
        """
        spatialDimensions = len(npArray.shape) - indexDimensions
        if spatialDimensions < 1:
            raise ValueError("Too many index dimensions. At least one spatial dimension required")

        fullLayout = getLayoutFromNumpyArray(npArray)
        spatialLayout = tuple([i for i in fullLayout if i < spatialDimensions])
        assert len(spatialLayout) == spatialDimensions

        strides = tuple([s // np.dtype(npArray.dtype).itemsize for s in npArray.strides])
        shape = tuple([int(s) for s in npArray.shape])

        return Field(fieldName, npArray.dtype, spatialLayout, shape, strides)

    @staticmethod
    def createGeneric(fieldName, spatialDimensions, dtype=np.float64, indexDimensions=0, layout=None):
        """
        Creates a generic field where the field size is not fixed i.e. can be called with arrays of different sizes
        :param fieldName: symbolic name for the field
        :param dtype: numpy data type of the array the kernel is called with later
        :param spatialDimensions: see documentation of Field
        :param indexDimensions: see documentation of Field
        :param layout: tuple specifying the loop ordering of the spatial dimensions e.g. (2, 1, 0 ) means that
                       the outer loop loops over dimension 2, the second outer over dimension 1, and the inner loop
                       over dimension 0
        """
        if not layout:
            layout = tuple(reversed(range(spatialDimensions)))
        if len(layout) != spatialDimensions:
            raise ValueError("Layout")
        shapeSymbol = IndexedBase(TypedSymbol(Field.SHAPE_PREFIX + fieldName, Field.SHAPE_DTYPE), shape=(1,))
        strideSymbol = IndexedBase(TypedSymbol(Field.STRIDE_PREFIX + fieldName, Field.STRIDE_DTYPE), shape=(1,))
        totalDimensions = spatialDimensions + indexDimensions
        shape = tuple([shapeSymbol[i] for i in range(totalDimensions)])
        strides = tuple([strideSymbol[i] for i in range(totalDimensions)])
        return Field(fieldName, dtype, layout, shape, strides)

    def __init__(self, fieldName, dtype, layout, shape, strides):
        """Do not use directly. Use static create* methods"""
        self._fieldName = fieldName
        self._dtype = numpyDataTypeToC(dtype)
        self._layout = layout
        self._shape = shape
        self._strides = strides
        self._readonly = False

    @property
    def spatialDimensions(self):
        return len(self._layout)

    @property
    def indexDimensions(self):
        return len(self._shape) - len(self._layout)

    @property
    def layout(self):
        return self._layout

    @property
    def name(self):
        return self._fieldName

    @property
    def shape(self):
        return self._shape

    @property
    def spatialShape(self):
        return self._shape[:self.spatialDimensions]

    @property
    def indexShape(self):
        return self._shape[self.spatialDimensions:]

    @property
    def spatialStrides(self):
        return self._strides[:self.spatialDimensions]

    @property
    def indexStrides(self):
        return self._strides[self.spatialDimensions:]

    @property
    def strides(self):
        return self._strides

    @property
    def dtype(self):
        return self._dtype

    @property
    def readOnly(self):
        return self._readonly

    def setReadOnly(self, value=True):
        self._readonly = value

    def __repr__(self):
        return self._fieldName

    def __getitem__(self, offset):
        if type(offset) is np.ndarray:
            offset = tuple(offset)
        if type(offset) is str:
            offset = tuple(directionStringToOffset(offset, self.spatialDimensions))
        if type(offset) is not tuple:
            offset = (offset,)
        if len(offset) != self.spatialDimensions:
            raise ValueError("Wrong number of spatial indices: "
                             "Got %d, expected %d" % (len(offset), self.spatialDimensions))
        return Field.Access(self, offset)

    def __call__(self, *args, **kwargs):
        center = tuple([0]*self.spatialDimensions)
        return Field.Access(self, center)(*args, **kwargs)

    def __hash__(self):
        return hash((self._layout, self._shape, self._strides, self._dtype, self._fieldName))

    def __eq__(self, other):
        selfTuple = (self.shape, self.strides, self.name, self.dtype)
        otherTuple = (other.shape, other.strides, other.name, other.dtype)
        return selfTuple == otherTuple

    PREFIX = "f"
    STRIDE_PREFIX = PREFIX + "stride_"
    SHAPE_PREFIX = PREFIX + "shape_"
    STRIDE_DTYPE = "const int *"
    SHAPE_DTYPE = "const int *"

    class Access(sp.Symbol):
        def __new__(cls, name, *args, **kwargs):
            obj = Field.Access.__xnew_cached_(cls, name, *args, **kwargs)
            return obj

        def __new_stage2__(self, field, offsets=(0, 0, 0), idx=None):
            fieldName = field.name
            offsetsAndIndex = chain(offsets, idx) if idx is not None else offsets
            constantOffsets = not any([isinstance(o, sp.Basic) for o in offsetsAndIndex])

            if not idx:
                idx = tuple([0] * field.indexDimensions)

            if constantOffsets:
                offsetName = offsetToDirectionString(offsets)

                if field.indexDimensions == 0:
                    obj = super(Field.Access, self).__xnew__(self, fieldName + "_" + offsetName)
                elif field.indexDimensions == 1:
                    obj = super(Field.Access, self).__xnew__(self, fieldName + "_" + offsetName + "^" + str(idx[0]))
                else:
                    idxStr = ",".join([str(e) for e in idx])
                    obj = super(Field.Access, self).__xnew__(self, fieldName + "_" + offsetName + "^" + idxStr)

            else:
                offsetName = "%0.10X" % (abs(hash(tuple(offsetsAndIndex))))
                obj = super(Field.Access, self).__xnew__(self, fieldName + "_" + offsetName)

            obj._field = field
            obj._offsets = []
            for o in offsets:
                if isinstance(o, sp.Basic):
                    obj._offsets.append(o)
                else:
                    obj._offsets.append(int(o))
            obj._offsetName = offsetName
            obj._index = idx

            return obj

        __xnew__ = staticmethod(__new_stage2__)
        __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

        def __call__(self, *idx):
            if self._index != tuple([0]*self.field.indexDimensions):
                print(self._index, tuple([0]*self.field.indexDimensions))
                raise ValueError("Indexing an already indexed Field.Access")

            idx = tuple(idx)
            if len(idx) != self.field.indexDimensions and idx != (0,):
                raise ValueError("Wrong number of indices: "
                                 "Got %d, expected %d" % (len(idx), self.field.indexDimensions))
            return Field.Access(self.field, self._offsets, idx)

        @property
        def field(self):
            return self._field

        @property
        def offsets(self):
            return self._offsets

        @property
        def requiredGhostLayers(self):
            return int(np.max(np.abs(self._offsets)))

        @property
        def nrOfCoordinates(self):
            return len(self._offsets)

        @property
        def offsetName(self):
            return self._offsetName

        @property
        def index(self):
            return self._index

        def _hashable_content(self):
            superClassContents = list(super(Field.Access, self)._hashable_content())
            t = tuple([*superClassContents, hash(self._field), self._index] + self._offsets)
            return t
