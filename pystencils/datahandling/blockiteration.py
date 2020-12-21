"""
This module contains function that simplify the iteration over walberla's distributed data structure.
These function simplify the iteration over rectangular slices, managing the mapping between block local coordinates and
global coordinates.
"""
import numpy as np

from pystencils.datahandling.datahandling_interface import Block
from pystencils.slicing import normalize_slice

try:
    # noinspection PyPep8Naming
    import waLBerla as wlb
except ImportError:
    wlb = None


def block_iteration(blocks, ghost_layers, dim=3, access_prefix=''):
    """Simple iteration over parallel walberla domain.

    Iterator that simplifies the access to field data by automatically converting from walberla fields to
    numpy arrays

    Args:
        blocks: walberla block data structure
        ghost_layers: how many ghost layers to include (outer and inner)
        dim: walberla's block data structure is 3D - 2D domains can be done by setting z_size=1
             if dim=2 is set here, the third coordinate of the returned fields is accessed at z=0 automatically
        access_prefix: see documentation of sliced_block_iteration
    """
    for block in blocks:
        cell_interval = blocks.getBlockCellBB(block)
        cell_interval.expand(ghost_layers)
        local_slice = [slice(0, w, None) for w in cell_interval.size]
        if dim == 2:
            local_slice[2] = ghost_layers
        yield ParallelBlock(block, cell_interval.min[:dim], tuple(local_slice), ghost_layers, access_prefix)


def sliced_block_iteration(blocks, slice_obj=None, inner_ghost_layers=1, outer_ghost_layers=1, dim=3, access_prefix=''):
    """Iterates of all blocks that have an intersection with the given slice object.

    For intersection blocks a Block object is yielded

    Args:
        blocks: walberla block data structure
        slice_obj: a slice (i.e. rectangular sub-region), can be created with make_slice[]
        inner_ghost_layers: how many ghost layers are included in the local slice and the optional index arrays
        outer_ghost_layers: slices can have relative coordinates e.g. make_slice[0.2, :, :]
                          when computing absolute values, the domain size is needed. This parameter
                          specifies how many ghost layers are taken into account for this operation.
        dim: set to 2 for pseudo 2D simulation (i.e. where z coordinate of blocks has extent 1)
             the arrays returned when indexing the block
        access_prefix: when accessing block data, this prefix is prepended to the access name
                      mostly used to switch between CPU and GPU field access (gpu fields are added with a
                      certain prefix 'gpu_')

    Example:
        assume no slice is given, then slice_normalization_ghost_layers effectively sets how much ghost layers at the
        border of the domain are included. The inner_ghost_layers parameter specifies how many inner ghost layers are
        included
    """
    if slice_obj is None:
        slice_obj = tuple([slice(None, None, None)] * dim)
    if dim == 2:
        slice_obj += (inner_ghost_layers,)

    domain_cell_bb = blocks.getDomainCellBB()
    domain_extent = [s + 2 * outer_ghost_layers for s in domain_cell_bb.size]
    slice_obj = normalize_slice(slice_obj, domain_extent)
    target_cell_bb = wlb.CellInterval.fromSlice(slice_obj)
    target_cell_bb.shift(*[a - outer_ghost_layers for a in domain_cell_bb.min])

    for block in blocks:
        intersection = blocks.getBlockCellBB(block).getExpanded(inner_ghost_layers)
        intersection.intersect(target_cell_bb)
        if intersection.empty():
            continue

        local_target_bb = blocks.transformGlobalToLocal(block, intersection)
        local_target_bb.shift(inner_ghost_layers, inner_ghost_layers, inner_ghost_layers)
        local_slice = local_target_bb.toSlice(False)
        if dim == 2:
            local_slice = (local_slice[0], local_slice[1], inner_ghost_layers)
        yield ParallelBlock(block, intersection.min[:dim], local_slice, inner_ghost_layers, access_prefix)


# ----------------------------- Implementation details -----------------------------------------------------------------


class SerialBlock(Block):
    """Simple mock-up block that is used for SerialDataHandling."""
    def __init__(self, field_dict, offset, local_slice):
        super(SerialBlock, self).__init__(offset, local_slice)
        self._fieldDict = field_dict

    def __getitem__(self, data_name):
        result = self._fieldDict[data_name]
        if isinstance(result, np.ndarray):
            result = result[self._localSlice]
        return result


class ParallelBlock(Block):
    def __init__(self, block, offset, local_slice, inner_ghost_layers, name_prefix):
        super(ParallelBlock, self).__init__(offset, local_slice)
        self._block = block
        self._gls = inner_ghost_layers
        self._name_prefix = name_prefix

    def __getitem__(self, data_name):
        result = self._block[self._name_prefix + data_name]
        type_name = type(result).__name__
        if 'GhostLayerField' in type_name:
            result = wlb.field.toArray(result, with_ghost_layers=self._gls)
            result = self._normalize_array_shape(result)
        elif 'GpuField' in type_name:
            result = wlb.cuda.toGpuArray(result, with_ghost_layers=self._gls)
            result = self._normalize_array_shape(result)
        return result

    def _normalize_array_shape(self, arr):
        if arr.shape[-1] == 1 and len(arr.shape) == 4:
            arr = arr[..., 0]
        return arr[self._localSlice]
