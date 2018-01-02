import numpy as np
import waLBerla as wlb
from pystencils.slicing import normalizeSlice


class BlockIterationInfo:
    def __init__(self, block, offset, localSlice):
        self._block = block
        self._offset = offset
        self._localSlice = localSlice

    @property
    def block(self):
        return self._block

    @property
    def offset(self):
        return self._offset

    @property
    def shape(self):
        return tuple(s.stop - s.start for s in self._localSlice)

    @property
    def localSlice(self):
        """Slice object of intersection between current block and iteration interval in local coordinates"""
        return self._localSlice

    @property
    def midpointArrays(self):
        """Global coordinate meshgrid of cell midpoints which are shifted by 0.5 compared to cell indices"""
        meshGridParams = [offset + 0.5 + np.arange(width, dtype=float)
                          for offset, width in zip(self.offset, self.shape)]
        return np.meshgrid(*meshGridParams, indexing='ij', copy=False)

    @property
    def cellIndexArrays(self):
        """Global coordinate meshgrid of cell coordinates. Cell indices start at 0 at the first inner cell,
        ghost layers have negative indices"""
        meshGridParams = [offset + np.arange(width, dtype=np.int32)
                          for offset, width in zip(self.offset, self.shape)]
        return np.meshgrid(*meshGridParams, indexing='ij', copy=False)


def slicedBlockIteration(blocks, sliceObj=None, innerGhostLayers=1, sliceNormalizationGhostLayers=1):
    """
    Iterates of all blocks that have an intersection with the given slice object.
    For these blocks a BlockIterationInfo object is yielded
    
    :param blocks: waLBerla block data structure
    :param sliceObj: a slice (i.e. rectangular subregion), can be created with makeSlice[]
    :param innerGhostLayers: how many ghost layers are included in the local slice and the optional index arrays
    :param sliceNormalizationGhostLayers: slices can have relative coordinates e.g. makeSlice[0.2, :, :]
                                          when computing absolute values, the domain size is needed. This parameter 
                                          specifies how many ghost layers are taken into account for this operation.

    Example: assume no slice is given, then sliceNormalizationGhostLayers effectively sets how much ghost layers
    at the border of the domain are included. The innerGhostLayers parameter specifies how many inner ghost layers are
    included
    """
    if sliceObj is None:
        sliceObj = [slice(None, None, None)] * 3

    domainCellBB = blocks.getDomainCellBB()
    domainExtent = [s + 2 * sliceNormalizationGhostLayers for s in domainCellBB.size]
    sliceObj = normalizeSlice(sliceObj, domainExtent)
    targetCellBB = wlb.CellInterval.fromSlice(sliceObj)
    targetCellBB.shift(*[a - sliceNormalizationGhostLayers for a in domainCellBB.min])

    for block in blocks:
        intersection = blocks.getBlockCellBB(block).getExpanded(innerGhostLayers)
        intersection.intersect(targetCellBB)
        if intersection.empty():
            continue

        localTargetBB = blocks.transformGlobalToLocal(block, intersection)
        localTargetBB.shift(innerGhostLayers, innerGhostLayers, innerGhostLayers)
        localSlice = localTargetBB.toSlice(False)
        yield BlockIterationInfo(block, intersection.min, localSlice)

