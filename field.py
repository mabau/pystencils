from enum import Enum
from itertools import chain
import numpy as np
import sympy as sp
from sympy.core.cache import cacheit
from sympy.tensor import IndexedBase

from pystencils.alignedarray import aligned_empty
from pystencils.data_types import TypedSymbol, createType, createCompositeTypeFromString, StructType
from pystencils.sympyextensions import isIntegerSequence


class FieldType(Enum):
    # generic fields
    GENERIC = 0
    # index fields are currently only used for boundary handling
    # the coordinates are not the loop counters in that case, but are read from this index field
    INDEXED = 1
    # communication buffer, used for (un)packing data in communication.
    BUFFER = 2

    @staticmethod
    def isGeneric(field):
        assert isinstance(field, Field)
        return field.fieldType == FieldType.GENERIC

    @staticmethod
    def isIndexed(field):
        assert isinstance(field, Field)
        return field.fieldType == FieldType.INDEXED

    @staticmethod
    def isBuffer(field):
        assert isinstance(field, Field)
        return field.fieldType == FieldType.BUFFER


class Field(object):
    """
    With fields one can formulate stencil-like update rules on structured grids.
    This Field class knows about the dimension, memory layout (strides) and optionally about the size of an array.

    Creating Fields:

        To create a field use one of the static create* members. There are two options:

        1. create a kernel with fixed loop sizes i.e. the shape of the array is already known. This is usually the
           case if just-in-time compilation directly from Python is done. (see :func:`Field.createFromNumpyArray`)
        2. create a more general kernel that works for variable array sizes. This can be used to create kernels
           beforehand for a library. (see :func:`Field.createGeneric`)

    Dimensions:
        A field has spatial and index dimensions, where the spatial dimensions come first.
        The interpretation is that the field has multiple cells in (usually) two or three dimensional space which are
        looped over. Additionally  N values are stored per cell. In this case spatialDimensions is two or three,
        and indexDimensions equals N. If you want to store a matrix on each point in a two dimensional grid, there
        are four dimensions, two spatial and two index dimensions: ``len(arr.shape) == spatialDims + indexDims``

    Indexing:
        When accessing (indexing) a field the result is a FieldAccess which is derived from sympy Symbol.
        First specify the spatial offsets in [], then in case indexDimension>0 the indices in ()
        e.g. ``f[-1,0,0](7)``

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
    def createGeneric(fieldName, spatialDimensions, dtype=np.float64, indexDimensions=0, layout='numpy',
                      indexShape=None, fieldType=FieldType.GENERIC):
        """
        Creates a generic field where the field size is not fixed i.e. can be called with arrays of different sizes

        :param fieldName: symbolic name for the field
        :param dtype: numpy data type of the array the kernel is called with later
        :param spatialDimensions: see documentation of Field
        :param indexDimensions: see documentation of Field
        :param layout: tuple specifying the loop ordering of the spatial dimensions e.g. (2, 1, 0 ) means that
                       the outer loop loops over dimension 2, the second outer over dimension 1, and the inner loop
                       over dimension 0. Also allowed: the strings 'numpy' (0,1,..d) or 'reverseNumpy' (d, ..., 1, 0)
        :param indexShape: optional shape of the index dimensions i.e. maximum values allowed for each index dimension,
                           has to be a list or tuple
        """
        if isinstance(layout, str):
            layout = spatialLayoutStringToTuple(layout, dim=spatialDimensions)
        shapeSymbol = IndexedBase(TypedSymbol(Field.SHAPE_PREFIX + fieldName, Field.SHAPE_DTYPE), shape=(1,))
        strideSymbol = IndexedBase(TypedSymbol(Field.STRIDE_PREFIX + fieldName, Field.STRIDE_DTYPE), shape=(1,))
        totalDimensions = spatialDimensions + indexDimensions
        if indexShape is None or len(indexShape) == 0:
            shape = tuple([shapeSymbol[i] for i in range(totalDimensions)])
        else:
            shape = tuple([shapeSymbol[i] for i in range(spatialDimensions)] + list(indexShape))

        strides = tuple([strideSymbol[i] for i in range(totalDimensions)])

        npDataType = np.dtype(dtype)
        if npDataType.fields is not None:
            if indexDimensions != 0:
                raise ValueError("Structured arrays/fields are not allowed to have an index dimension")
            shape += (1,)
            strides += (1,)

        return Field(fieldName, fieldType, dtype, layout, shape, strides)

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

        fullLayout = getLayoutOfArray(npArray)
        spatialLayout = tuple([i for i in fullLayout if i < spatialDimensions])
        assert len(spatialLayout) == spatialDimensions

        strides = tuple([s // np.dtype(npArray.dtype).itemsize for s in npArray.strides])
        shape = tuple(int(s) for s in npArray.shape)

        npDataType = np.dtype(npArray.dtype)
        if npDataType.fields is not None:
            if indexDimensions != 0:
                raise ValueError("Structured arrays/fields are not allowed to have an index dimension")
            shape += (1,)
            strides += (1,)

        return Field(fieldName, FieldType.GENERIC, npArray.dtype, spatialLayout, shape, strides)

    @staticmethod
    def createFixedSize(fieldName, shape, indexDimensions=0, dtype=np.float64, layout='numpy', strides=None):
        """
        Creates a field with fixed sizes i.e. can be called only with arrays of the same size and layout

        :param fieldName: symbolic name for the field
        :param shape: overall shape of the array
        :param indexDimensions: how many of the trailing dimensions are interpreted as index (as opposed to spatial)
        :param dtype: numpy data type of the array the kernel is called with later
        :param layout: full layout of array, not only spatial dimensions
        :param strides: strides in bytes or None to automatically compute them from shape (assuming no padding)
        """
        spatialDimensions = len(shape) - indexDimensions
        assert spatialDimensions >= 1

        if isinstance(layout, str):
            layout = layoutStringToTuple(layout, spatialDimensions + indexDimensions)

        shape = tuple(int(s) for s in shape)
        if strides is None:
            strides = computeStrides(shape, layout)
        else:
            assert len(strides) == len(shape)
            strides = tuple([s // np.dtype(dtype).itemsize for s in strides])

        npDataType = np.dtype(dtype)
        if npDataType.fields is not None:
            if indexDimensions != 0:
                raise ValueError("Structured arrays/fields are not allowed to have an index dimension")
            shape += (1,)
            strides += (1,)

        spatialLayout = list(layout)
        for i in range(spatialDimensions, len(layout)):
            spatialLayout.remove(i)
        return Field(fieldName, FieldType.GENERIC, dtype, tuple(spatialLayout), shape, strides)

    def __init__(self, fieldName, fieldType, dtype, layout, shape, strides):
        """Do not use directly. Use static create* methods"""
        self._fieldName = fieldName
        assert isinstance(fieldType, FieldType)
        self.fieldType = fieldType
        self._dtype = createType(dtype)
        self._layout = normalizeLayout(layout)
        self.shape = shape
        self.strides = strides
        self.latexName = None

    def newFieldWithDifferentName(self, newName):
        return Field(newName, self.fieldType, self._dtype, self._layout, self.shape, self.strides)

    @property
    def spatialDimensions(self):
        return len(self._layout)

    @property
    def indexDimensions(self):
        return len(self.shape) - len(self._layout)

    @property
    def layout(self):
        return self._layout

    @property
    def name(self):
        return self._fieldName

    @property
    def spatialShape(self):
        return self.shape[:self.spatialDimensions]

    @property
    def indexShape(self):
        return self.shape[self.spatialDimensions:]

    @property
    def hasFixedShape(self):
        return isIntegerSequence(self.shape)

    @property
    def indexShape(self):
        return self.shape[self.spatialDimensions:]

    @property
    def hasFixedIndexShape(self):
        return isIntegerSequence(self.indexShape)

    @property
    def spatialStrides(self):
        return self.strides[:self.spatialDimensions]

    @property
    def indexStrides(self):
        return self.strides[self.spatialDimensions:]

    @property
    def dtype(self):
        return self._dtype

    def __repr__(self):
        return self._fieldName

    def neighbor(self, coordId, offset):
        offsetList = [0] * self.spatialDimensions
        offsetList[coordId] = offset
        return Field.Access(self, tuple(offsetList))

    def neighbors(self, stencil):
        return [self.__getitem__(s) for s in stencil]

    @property
    def vecCenter(self):
        indexShape = self.indexShape
        if len(indexShape) == 0:
            return self.center
        elif len(indexShape) == 1:
            return sp.Matrix([self(i) for i in range(indexShape[0])])
        elif len(indexShape) == 2:
            def cb(*args):
                r = self.__call__(*args)
                return r
            return sp.Matrix(*indexShape, cb)

    @property
    def center(self):
        center = tuple([0] * self.spatialDimensions)
        return Field.Access(self, center)

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
        return hash((self._layout, self.shape, self.strides, self._dtype, self.fieldType, self._fieldName))

    def __eq__(self, other):
        selfTuple = (self.shape, self.strides, self.name, self.dtype, self.fieldType)
        otherTuple = (other.shape, other.strides, other.name, other.dtype, other.fieldType)
        return selfTuple == otherTuple


    PREFIX = "f"
    STRIDE_PREFIX = PREFIX + "stride_"
    SHAPE_PREFIX = PREFIX + "shape_"
    STRIDE_DTYPE = createCompositeTypeFromString("const int *")
    SHAPE_DTYPE = createCompositeTypeFromString("const int *")
    DATA_PREFIX = PREFIX + "d_"

    class Access(sp.Symbol):
        def __new__(cls, name, *args, **kwargs):
            obj = Field.Access.__xnew_cached_(cls, name, *args, **kwargs)
            return obj

        def __new_stage2__(self, field, offsets=(0, 0, 0), idx=None):
            fieldName = field.name
            offsetsAndIndex = chain(offsets, idx) if idx is not None else offsets
            constantOffsets = not any([isinstance(o, sp.Basic) and not o.is_Integer for o in offsetsAndIndex])

            if not idx:
                idx = tuple([0] * field.indexDimensions)

            if constantOffsets:
                offsetName = offsetToDirectionString(offsets)
                if field.indexDimensions == 0:
                    superscript = None
                elif field.indexDimensions == 1:
                    superscript = str(idx[0])
                else:
                    idxStr = ",".join([str(e) for e in idx])
                    superscript = idxStr
                if field.hasFixedIndexShape and not isinstance(field.dtype, StructType):
                    for i, bound in zip(idx, field.indexShape):
                        if i >= bound:
                            raise ValueError("Field index out of bounds")
            else:
                offsetName = "%0.10X" % (abs(hash(tuple(offsetsAndIndex))))
                superscript = None

            symbolName = "%s_%s" % (fieldName, offsetName)
            if superscript is not None:
                symbolName += "^" + superscript

            obj = super(Field.Access, self).__xnew__(self, symbolName)
            obj._field = field
            obj._offsets = []
            for o in offsets:
                if isinstance(o, sp.Basic):
                    obj._offsets.append(o)
                else:
                    obj._offsets.append(int(o))
            obj._offsetName = offsetName
            obj._superscript = superscript
            obj._index = idx

            return obj

        def __getnewargs__(self):
            return self.field, self.offsets, self.index

        __xnew__ = staticmethod(__new_stage2__)
        __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

        def __call__(self, *idx):
            if self._index != tuple([0]*self.field.indexDimensions):
                raise ValueError("Indexing an already indexed Field.Access")

            idx = tuple(idx)

            if self.field.indexDimensions == 0 and idx == (0,):
                idx = ()

            if len(idx) != self.field.indexDimensions:
                raise ValueError("Wrong number of indices: "
                                 "Got %d, expected %d" % (len(idx), self.field.indexDimensions))
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
        def requiredGhostLayers(self):
            return int(np.max(np.abs(self._offsets)))

        @property
        def nrOfCoordinates(self):
            return len(self._offsets)

        @property
        def offsetName(self):
            return self._offsetName

        def _latex(self, arg):
            n = self._field.latexName if self._field.latexName else self._field.name
            if self._superscript:
                return "{{%s}_{%s}^{%s}}" % (n, self._offsetName, self._superscript)
            else:
                return "{{%s}_{%s}}" % (n, self._offsetName)

        @property
        def index(self):
            return self._index

        def getNeighbor(self, *offsets):
            return Field.Access(self.field, offsets, self.index)

        def neighbor(self, coordId, offset):
            offsetList = list(self.offsets)
            offsetList[coordId] += offset
            return Field.Access(self.field, tuple(offsetList), self.index)

        def getShifted(self, *shift):
            return Field.Access(self.field, tuple(a + b for a, b in zip(shift, self.offsets)), self.index)

        def _hashable_content(self):
            superClassContents = list(super(Field.Access, self)._hashable_content())
            t = tuple(superClassContents + [hash(self._field), self._index] + self._offsets)
            return t


def extractCommonSubexpressions(equations):
    """
    Uses sympy to find common subexpressions in equations and returns
    them in a topologically sorted order, ready for evaluation.
    Usually called before list of equations is passed to :func:`createKernel`
    """
    replacements, newEq = sp.cse(equations)
    # Workaround for older sympy versions: here subexpressions (temporary = True) are extracted
    # which leads to problems in Piecewise functions which have to a default case indicated by True
    symbolsEqualToTrue = {r[0]: True for r in replacements if r[1] is sp.true}

    replacementEqs = [sp.Eq(*r) for r in replacements if r[1] is not sp.true]
    equations = replacementEqs + newEq
    topologicallySortedPairs = sp.cse_main.reps_toposort([[e.lhs, e.rhs] for e in equations])
    equations = [sp.Eq(a[0], a[1].subs(symbolsEqualToTrue)) for a in topologicallySortedPairs]
    return equations


def getLayoutFromStrides(strides, indexDimensionIds=[]):
    coordinates = list(range(len(strides)))
    relevantStrides = [stride for i, stride in enumerate(strides) if i not in indexDimensionIds]
    result = [x for (y, x) in sorted(zip(relevantStrides, coordinates), key=lambda pair: pair[0], reverse=True)]
    return normalizeLayout(result)


def getLayoutOfArray(arr, indexDimensionIds=[]):
    """
    Returns a list indicating the memory layout (linearization order) of the numpy array.
    Example:
    >>> getLayoutOfArray(np.zeros([3,3,3]))
    (0, 1, 2)

    In this example the loop over the zeroth coordinate should be the outermost loop,
    followed by the first and second. Elements arr[x,y,0] and arr[x,y,1] are adjacent in memory.
    Normally constructed numpy arrays have this order, however by stride tricks or other frameworks, arrays
    with different memory layout can be created.

    The indexDimensionIds parameter leaves specifies which coordinates should not be
    """
    return getLayoutFromStrides(arr.strides, indexDimensionIds)


def createNumpyArrayWithLayout(shape, layout, alignment=False, byteOffset=0, **kwargs):
    """
    Creates a numpy array with
    :param shape: shape of the resulting array
    :param layout: layout as tuple, where the coordinates are ordered from slow to fast
    :param alignment: number of bytes to align the beginning and the innermost coordinate to, or False for no alignment
    :param byteOffset: only used when alignment is specified, align not beginning but address at this offset
                       mostly used to align first inner cell, not ghost cells
    >>> res = createNumpyArrayWithLayout(shape=(2, 3, 4, 5), layout=(3, 2, 0, 1))
    >>> res.shape
    (2, 3, 4, 5)
    >>> getLayoutOfArray(res)
    (3, 2, 0, 1)
    """
    assert set(layout) == set(range(len(shape))), "Wrong layout descriptor"
    currentLayout = list(range(len(shape)))
    swaps = []
    for i in range(len(layout)):
        if currentLayout[i] != layout[i]:
            indexToSwapWith = currentLayout.index(layout[i])
            swaps.append((i, indexToSwapWith))
            currentLayout[i], currentLayout[indexToSwapWith] = currentLayout[indexToSwapWith], currentLayout[i]
    assert tuple(currentLayout) == tuple(layout)

    shape = list(shape)
    for a, b in swaps:
        shape[a], shape[b] = shape[b], shape[a]

    if not alignment:
        res = np.empty(shape, order='c', **kwargs)
    else:
        if alignment is True:
            alignment = 8 * 4
        res = aligned_empty(shape, alignment, byteOffset=byteOffset, **kwargs)

    for a, b in reversed(swaps):
        res = res.swapaxes(a, b)
    return res


def spatialLayoutStringToTuple(layoutStr, dim):
    if layoutStr in ('fzyx', 'zyxf'):
        assert dim <= 3
        return tuple(reversed(range(dim)))

    if layoutStr in ('fzyx', 'f', 'reverseNumpy', 'SoA'):
        return tuple(reversed(range(dim)))
    elif layoutStr in ('c', 'numpy', 'AoS'):
        return tuple(range(dim))
    raise ValueError("Unknown layout descriptor " + layoutStr)


def layoutStringToTuple(layoutStr, dim):
    layoutStr = layoutStr.lower()
    if layoutStr == 'fzyx' or layoutStr == 'soa':
        assert dim <= 4
        return tuple(reversed(range(dim)))
    elif layoutStr == 'zyxf' or layoutStr == 'aos':
        assert dim <= 4
        return tuple(reversed(range(dim - 1))) + (dim-1,)
    elif layoutStr == 'f' or layoutStr == 'reversenumpy':
        return tuple(reversed(range(dim)))
    elif layoutStr == 'c' or layoutStr == 'numpy':
        return tuple(range(dim))
    raise ValueError("Unknown layout descriptor " + layoutStr)


def normalizeLayout(layout):
    """Takes a layout tuple and subtracts the minimum from all entries"""
    minEntry = min(layout)
    return tuple(i - minEntry for i in layout)


def computeStrides(shape, layout):
    """
    Computes strides assuming no padding exists
    :param shape: shape (size) of array
    :param layout: layout specification as tuple
    :return: strides in elements, not in bytes
    """
    N = len(shape)
    assert len(layout) == N
    assert len(set(layout)) == N
    strides = [0] * N
    product = 1
    for j in reversed(layout):
        strides[j] = product
        product *= shape[j]
    return tuple(strides)


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


if __name__ == '__main__':
    f = Field.createGeneric('f', spatialDimensions=2, indexShape=(2,4))
    f(2, 0)
    fa = f[0, 1](4) ** 2
    print(fa)
    print(sp.latex(fa))