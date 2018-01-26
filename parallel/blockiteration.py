"""
This module contains function that simplify the iteration over waLBerlas distributed data structure.
These function simplify the iteration over rectangular slices, managing the mapping between block local coordinates and
global coordinates.
"""
import numpy as np
import waLBerla as wlb
from pystencils.slicing import normalizeSlice


def blockIteration(blocks, ghostLayers, dim=3, accessPrefix=''):
    """
    Iterator that simplifies the access to field data by automatically converting from waLBerla fields to
    numpy arrays
    :param blocks: waLBerla block data structure
    :param ghostLayers: how many ghost layers to include (outer and inner)
    :param dim: waLBerlas block data structure is 3D - 2D domains can be done by setting zSize=1
                if dim=2 is set here, the third coordinate of the returned fields is accessed at z=0 automatically
    :param accessPrefix: see documentation of slicedBlockIteration
    """
    for block in blocks:
        cellInterval = blocks.getBlockCellBB(block)
        cellInterval.expand(ghostLayers)
        localSlice = [slice(0, w, None) for w in cellInterval.size]
        if dim == 2:
            localSlice[2] = ghostLayers
        yield ParallelBlock(block, cellInterval.min[:dim], tuple(localSlice), ghostLayers, accessPrefix)


def slicedBlockIteration(blocks, sliceObj=None, innerGhostLayers=1, outerGhostLayers=1, dim=3, accessPrefix=''):
    """
    Iterates of all blocks that have an intersection with the given slice object.
    For these blocks a Block object is yielded
    
    :param blocks: waLBerla block data structure
    :param sliceObj: a slice (i.e. rectangular sub-region), can be created with makeSlice[]
    :param innerGhostLayers: how many ghost layers are included in the local slice and the optional index arrays
    :param outerGhostLayers: slices can have relative coordinates e.g. makeSlice[0.2, :, :]
                             when computing absolute values, the domain size is needed. This parameter
                             specifies how many ghost layers are taken into account for this operation.
    :param dim: set to 2 for pseudo 2D simulation (i.e. where z coordinate of blocks has extent 1)
                the arrays returned when indexing the block
    :param accessPrefix: when accessing block data, this prefix is prepended to the access name
                         mostly used to switch between CPU and GPU field access (gpu fields are added with a
                         certain prefix 'gpu_')
    Example: assume no slice is given, then sliceNormalizationGhostLayers effectively sets how much ghost layers
    at the border of the domain are included. The innerGhostLayers parameter specifies how many inner ghost layers are
    included
    """
    if sliceObj is None:
        sliceObj = tuple([slice(None, None, None)] * dim)
    if dim == 2:
        sliceObj += (innerGhostLayers, )

    domainCellBB = blocks.getDomainCellBB()
    domainExtent = [s + 2 * outerGhostLayers for s in domainCellBB.size]
    sliceObj = normalizeSlice(sliceObj, domainExtent)
    targetCellBB = wlb.CellInterval.fromSlice(sliceObj)
    targetCellBB.shift(*[a - outerGhostLayers for a in domainCellBB.min])

    for block in blocks:
        intersection = blocks.getBlockCellBB(block).getExpanded(innerGhostLayers)
        intersection.intersect(targetCellBB)
        if intersection.empty():
            continue

        localTargetBB = blocks.transformGlobalToLocal(block, intersection)
        localTargetBB.shift(innerGhostLayers, innerGhostLayers, innerGhostLayers)
        localSlice = localTargetBB.toSlice(False)
        if dim == 2:
            localSlice = (localSlice[0], localSlice[1], innerGhostLayers)
        yield ParallelBlock(block, intersection.min[:dim], localSlice, innerGhostLayers, accessPrefix)


class Block:
    def __init__(self, offset, localSlice):
        self._offset = offset
        self._localSlice = localSlice

    @property
    def offset(self):
        """Offset of the current block in global coordinates (where lower ghost layers have negative indices)"""
        return self._offset

    @property
    def cellIndexArrays(self):
        """Global coordinate meshgrid of cell coordinates. Cell indices start at 0 at the first inner cell,
        lower ghost layers have negative indices"""
        meshGridParams = [offset + np.arange(width, dtype=np.int32)
                          for offset, width in zip(self.offset, self.shape)]
        return np.meshgrid(*meshGridParams, indexing='ij', copy=False)

    @property
    def midpointArrays(self):
        """Global coordinate meshgrid of cell midpoints which are shifted by 0.5 compared to cell indices"""
        meshGridParams = [offset + 0.5 + np.arange(width, dtype=float)
                          for offset, width in zip(self.offset, self.shape)]
        return np.meshgrid(*meshGridParams, indexing='ij', copy=False)

    @property
    def shape(self):
        """Shape of the fields (potentially including ghost layers)"""
        return tuple(s.stop - s.start for s in self._localSlice[:len(self._offset)])

    @property
    def globalSlice(self):
        """Slice in global coordinates"""
        return tuple(slice(off, off+size) for off, size in zip(self._offset, self.shape))

# ----------------------------- Implementation details -----------------------------------------------------------------


class SerialBlock(Block):
    """Simple mockup block that is used if SerialDataHandling."""
    def __init__(self, fieldDict, offset, localSlice):
        super(SerialBlock, self).__init__(offset, localSlice)
        self._fieldDict = fieldDict

    def __getitem__(self, dataName):
        result = self._fieldDict[dataName]
        if isinstance(result, np.ndarray):
            result = result[self._localSlice]
        return result


class ParallelBlock(Block):
    def __init__(self, block, offset, localSlice, innerGhostLayers, namePrefix):
        super(ParallelBlock, self).__init__(offset, localSlice)
        self._block = block
        self._gls = innerGhostLayers
        self._namePrefix = namePrefix

    def __getitem__(self, dataName):
        result = self._block[self._namePrefix + dataName]
        typeName = type(result).__name__
        if typeName == 'GhostLayerField':
            result = wlb.field.toArray(result, withGhostLayers=self._gls)
            result = self._normalizeArrayShape(result)
        elif typeName == 'GpuField':
            result = wlb.cuda.toGpuArray(result, withGhostLayers=self._gls)
            result = self._normalizeArrayShape(result)
        return result

    def _normalizeArrayShape(self, arr):
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr[self._localSlice]

