import sympy as sp
import numpy as np


class SliceMaker(object):
    def __getitem__(self, item):
        return item
makeSlice = SliceMaker()


def normalizeSlice(slices, sizes):
    """Converts slices with floating point and/or negative entries to integer slices"""

    if len(slices) != len(sizes):
        raise ValueError("Slice dimension does not match sizes")

    result = []

    for s, size in zip(slices, sizes):
        if type(s) is int:
            result.append(s)
            continue
        if type(s) is float:
            result.append(int(s * size))
            continue

        assert (type(s) is slice)

        if s.start is None:
            newStart = 0
        elif type(s.start) is float:
            newStart = int(s.start * size)
        elif not isinstance(s.start, sp.Basic) and s.start < 0:
            newStart = size + s.start
        else:
            newStart = s.start

        if s.stop is None:
            newStop = size
        elif type(s.stop) is float:
            newStop = int(s.stop * size)
        elif not isinstance(s.stop, sp.Basic) and s.stop < 0:
            newStop = size + s.stop
        else:
            newStop = s.stop

        result.append(slice(newStart, newStop, s.step if s.step is not None else 1))

    return tuple(result)


def shiftSlice(slices, offset):
    return [slice(k.start+offset, k.stop + offset, k.step) for k in slices]


def sliceFromDirection(directionName, dim, normalOffset=0, tangentialOffset=0):
    """
    Create a slice from a direction named by compass scheme:
        i.e. 'N' for north returns same as makeSlice[:, -1]
        the naming is:
            - x: W, E (west, east)
            - y: S, N (south, north)
            - z: B, T (bottom, top)
    Also combinations are allowed like north-east 'NE'

    :param directionName: name of direction as explained above
    :param dim: dimension of the returned slice (should be 2 or 3)
    :param normalOffset: the offset in 'normal' direction: e.g. sliceFromDirection('N',2, normalOffset=2)
                         would return makeSlice[:, -3]
    :param tangentialOffset: offset in the other directions: e.g. sliceFromDirection('N',2, tangentialOffset=2)
                         would return makeSlice[2:-2, -1]
    """
    if tangentialOffset == 0:
        result = [slice(None, None, None)] * dim
    else:
        result = [slice(tangentialOffset, -tangentialOffset, None)] * dim

    normalSliceHigh, normalSliceLow = -1-normalOffset, normalOffset

    for dimIdx, (lowName, highName) in enumerate([('W', 'E'), ('S', 'N'), ('B', 'T')]):
        if lowName in directionName:
            assert highName not in directionName, "Invalid direction name"
            result[dimIdx] = normalSliceLow
        if highName in directionName:
            assert lowName not in directionName, "Invalid direction name"
            result[dimIdx] = normalSliceHigh
    return tuple(result)


def removeGhostLayers(arr, indexDimensions=0, ghostLayers=1):
    dimensions = len(arr.shape)
    spatialDimensions = dimensions - indexDimensions
    indexing = [slice(ghostLayers, -ghostLayers, None), ] * spatialDimensions
    indexing += [slice(None, None, None)] * indexDimensions
    return arr[indexing]


def addGhostLayers(arr, indexDimensions=0, ghostLayers=1):
    dimensions = len(arr.shape)
    spatialDimensions = dimensions - indexDimensions
    newShape = [e + 2 * ghostLayers for e in arr.shape[:spatialDimensions]] + list(arr.shape[spatialDimensions:])
    result = np.zeros(newShape)
    indexing = [slice(ghostLayers, -ghostLayers, None), ] * spatialDimensions
    indexing += [slice(None, None, None)] * indexDimensions
    result[indexing] = arr
    return result


def getSliceBeforeGhostLayer(direction, ghostLayers=1, thickness=None, fullSlice=False):
    """
    Returns slicing expression for region before ghost layer
    :param direction: tuple specifying direction of slice
    :param ghostLayers: number of ghost layers
    :param thickness: thickness of the slice, defaults to number of ghost layers
    :param fullSlice:  if true also the ghost cells in directions orthogonal to direction are contained in the
                       returned slice. Example (d=W ): if fullSlice then also the ghost layer in N-S and T-B
                       are included, otherwise only inner cells are returned
    """
    if not thickness:
        thickness = ghostLayers
    fullSliceInc = ghostLayers if not fullSlice else 0
    slices = []
    for dirComponent in direction:
        if dirComponent == -1:
            s = slice(ghostLayers, thickness + ghostLayers)
        elif dirComponent == 0:
            end = -fullSliceInc
            s = slice(fullSliceInc, end if end != 0 else None)
        elif dirComponent == 1:
            start = -thickness - ghostLayers
            end = -ghostLayers
            s = slice(start if start != 0 else None, end if end != 0 else None)
        else:
            raise ValueError("Invalid direction: only -1, 0, 1 components are allowed")
        slices.append(s)
    return tuple(slices)


def getGhostRegionSlice(direction, ghostLayers=1, thickness=None, fullSlice=False):
    """
    Returns slice of ghost region. For parameters see :func:`getSliceBeforeGhostLayer`
    """
    if not thickness:
        thickness = ghostLayers
    assert thickness > 0
    assert thickness <= ghostLayers
    fullSliceInc = ghostLayers if not fullSlice else 0
    slices = []
    for dirComponent in direction:
        if dirComponent == -1:
            s = slice(ghostLayers - thickness, ghostLayers)
        elif dirComponent == 0:
            end = -fullSliceInc
            s = slice(fullSliceInc, end if end != 0 else None)
        elif dirComponent == 1:
            start = -ghostLayers
            end = - ghostLayers + thickness
            s = slice(start if start != 0 else None, end if end != 0 else None)
        else:
            raise ValueError("Invalid direction: only -1, 0, 1 components are allowed")
        slices.append(s)
    return tuple(slices)


def getPeriodicBoundarySrcDstSlices(stencil, ghostLayers=1, thickness=None):
    srcDstSliceTuples = []

    for d in stencil:
        if sum([abs(e) for e in d]) == 0:
            continue
        invDir = (-e for e in d)
        src = getSliceBeforeGhostLayer(invDir, ghostLayers, thickness=thickness, fullSlice=False)
        dst = getGhostRegionSlice(d, ghostLayers, thickness=thickness, fullSlice=False)
        srcDstSliceTuples.append((src, dst))
    return srcDstSliceTuples


def getPeriodicBoundaryFunctor(stencil, ghostLayers=1, thickness=None):
    """
    Returns a function that applies periodic boundary conditions
    :param stencil: sequence of directions e.g. ( [0,1], [0,-1] ) for y periodicity
    :param ghostLayers: how many ghost layers the array has
    :param thickness: how many of the ghost layers to copy, None means 'all'
    :return: function that takes a single array and applies the periodic copy operation
    """
    srcDstSliceTuples = getPeriodicBoundarySrcDstSlices(stencil, ghostLayers, thickness)

    def functor(pdfs, **kwargs):
        for srcSlice, dstSlice in srcDstSliceTuples:
            pdfs[dstSlice] = pdfs[srcSlice]

    return functor


