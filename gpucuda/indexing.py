import abc
from typing import Tuple  # noqa
import sympy as sp
from pystencils.astnodes import Conditional, Block
from pystencils.integer_functions import div_ceil
from pystencils.slicing import normalize_slice
from pystencils.data_types import TypedSymbol, create_type
from functools import partial

AUTO_BLOCK_SIZE_LIMITING = False

BLOCK_IDX = [TypedSymbol("blockIdx." + coord, create_type("int")) for coord in ('x', 'y', 'z')]
THREAD_IDX = [TypedSymbol("threadIdx." + coord, create_type("int")) for coord in ('x', 'y', 'z')]
BLOCK_DIM = [TypedSymbol("blockDim." + coord, create_type("int")) for coord in ('x', 'y', 'z')]
GRID_DIM = [TypedSymbol("gridDim." + coord, create_type("int")) for coord in ('x', 'y', 'z')]


class AbstractIndexing(abc.ABC):
    """
    Abstract base class for all Indexing classes. An Indexing class defines how a multidimensional
    field is mapped to CUDA's block and grid system. It calculates indices based on CUDA's thread and block indices
    and computes the number of blocks and threads a kernel is started with. The Indexing class is created with
    a pystencils field, a slice to iterate over, and further optional parameters that must have default values.
    """

    @property
    @abc.abstractmethod
    def coordinates(self):
        """Returns a sequence of coordinate expressions for (x,y,z) depending on symbolic CUDA block and thread indices.
        These symbolic indices can be obtained with the method `index_variables` """

    @property
    def index_variables(self):
        """Sympy symbols for CUDA's block and thread indices, and block and grid dimensions. """
        return BLOCK_IDX + THREAD_IDX + BLOCK_DIM + GRID_DIM

    @abc.abstractmethod
    def call_parameters(self, arr_shape):
        """Determine grid and block size for kernel call.

        Args:
            arr_shape: the numeric (not symbolic) shape of the array
        Returns:
            dict with keys 'blocks' and 'threads' with tuple values for number of (x,y,z) threads and blocks
            the kernel should be started with
        """

    @abc.abstractmethod
    def guard(self, kernel_content, arr_shape):
        """In some indexing schemes not all threads of a block execute the kernel content.

        This function can return a Conditional ast node, defining this execution guard.

        Args:
            kernel_content: the actual kernel contents which can e.g. be put into the Conditional node as true block
            arr_shape: the numeric or symbolic shape of the field

        Returns:
            ast node, which is put inside the kernel function
        """


# -------------------------------------------- Implementations ---------------------------------------------------------


class BlockIndexing(AbstractIndexing):
    """Generic indexing scheme that maps sub-blocks of an array to CUDA blocks.

    Args:
        field: pystencils field (common to all Indexing classes)
        iteration_slice: slice that defines rectangular subarea which is iterated over
        permute_block_size_dependent_on_layout: if True the block_size is permuted such that the fastest coordinate
                                                gets the largest amount of threads
        compile_time_block_size: compile in concrete block size, otherwise the cuda variable 'blockDim' is used
    """
    def __init__(self, field, iteration_slice=None,
                 block_size=(16, 16, 1), permute_block_size_dependent_on_layout=True, compile_time_block_size=False):
        if field.spatial_dimensions > 3:
            raise NotImplementedError("This indexing scheme supports at most 3 spatial dimensions")

        if permute_block_size_dependent_on_layout:
            block_size = self.permute_block_size_according_to_layout(block_size, field.layout)

        if AUTO_BLOCK_SIZE_LIMITING:
            block_size = self.limit_block_size_to_device_maximum(block_size)

        self._block_size = block_size
        self._iterationSlice = normalize_slice(iteration_slice, field.spatial_shape)
        self._dim = field.spatial_dimensions
        self._symbolic_shape = [e if isinstance(e, sp.Basic) else None for e in field.spatial_shape]
        self._compile_time_block_size = compile_time_block_size

    @property
    def coordinates(self):
        offsets = _get_start_from_slice(self._iterationSlice)
        block_size = self._block_size if self._compile_time_block_size else BLOCK_DIM
        coordinates = [block_index * bs + thread_idx + off
                       for block_index, bs, thread_idx, off in zip(BLOCK_IDX, block_size, THREAD_IDX, offsets)]

        return coordinates[:self._dim]

    def call_parameters(self, arr_shape):
        substitution_dict = {sym: value for sym, value in zip(self._symbolic_shape, arr_shape) if sym is not None}

        widths = [end - start for start, end in zip(_get_start_from_slice(self._iterationSlice),
                                                    _get_end_from_slice(self._iterationSlice, arr_shape))]
        widths = sp.Matrix(widths).subs(substitution_dict)
        extend_bs = (1,) * (3 - len(self._block_size))
        block_size = self._block_size + extend_bs
        if not self._compile_time_block_size:
            block_size = tuple(sp.Min(bs, shape) for bs, shape in zip(block_size, widths)) + extend_bs

        grid = tuple(div_ceil(length, block_size)
                     for length, block_size in zip(widths, block_size))
        extend_gr = (1,) * (3 - len(grid))

        return {'block': block_size,
                'grid': grid + extend_gr}

    def guard(self, kernel_content, arr_shape):
        arr_shape = arr_shape[:self._dim]
        conditions = [c < end
                      for c, end in zip(self.coordinates, _get_end_from_slice(self._iterationSlice, arr_shape))]
        condition = conditions[0]
        for c in conditions[1:]:
            condition = sp.And(condition, c)
        return Block([Conditional(condition, kernel_content)])

    @staticmethod
    def limit_block_size_to_device_maximum(block_size):
        """Changes block size according to match device limits.

        * if the total amount of threads is too big for the current device, the biggest coordinate is divided by 2.
        * next, if one component is still too big, the component which is too big is divided by 2 and the smallest
          component is multiplied by 2, such that the total amount of threads stays the same

        Returns:
            the altered block_size
        """
        # Get device limits
        import pycuda.driver as cuda
        # noinspection PyUnresolvedReferences
        import pycuda.autoinit  # NOQA

        da = cuda.device_attribute
        device = cuda.Context.get_device()

        block_size = list(block_size)
        max_threads = device.get_attribute(da.MAX_THREADS_PER_BLOCK)
        max_block_size = [device.get_attribute(a)
                          for a in (da.MAX_BLOCK_DIM_X, da.MAX_BLOCK_DIM_Y, da.MAX_BLOCK_DIM_Z)]

        def prod(seq):
            result = 1
            for e in seq:
                result *= e
            return result

        def get_index_of_too_big_element():
            for i, bs in enumerate(block_size):
                if bs > max_block_size[i]:
                    return i
            return None

        def get_index_of_too_small_element():
            for i, bs in enumerate(block_size):
                if bs // 2 <= max_block_size[i]:
                    return i
            return None

        # Reduce the total number of threads if necessary
        while prod(block_size) > max_threads:
            item_to_reduce = block_size.index(max(block_size))
            for j, block_size_entry in enumerate(block_size):
                if block_size_entry > max_block_size[j]:
                    item_to_reduce = j
            block_size[item_to_reduce] //= 2

        # Cap individual elements
        too_big_element_index = get_index_of_too_big_element()
        while too_big_element_index is not None:
            too_small_element_index = get_index_of_too_small_element()
            block_size[too_small_element_index] *= 2
            block_size[too_big_element_index] //= 2
            too_big_element_index = get_index_of_too_big_element()

        return tuple(block_size)

    @staticmethod
    def limit_block_size_by_register_restriction(block_size, required_registers_per_thread, device=None):
        """Shrinks the block_size if there are too many registers used per multiprocessor.
        This is not done automatically, since the required_registers_per_thread are not known before compilation.
        They can be obtained by ``func.num_regs`` from a pycuda function.
        :returns smaller block_size if too many registers are used.
        """
        import pycuda.driver as cuda
        # noinspection PyUnresolvedReferences
        import pycuda.autoinit  # NOQA

        da = cuda.device_attribute
        if device is None:
            device = cuda.Context.get_device()
        available_registers_per_mp = device.get_attribute(da.MAX_REGISTERS_PER_MULTIPROCESSOR)

        block = block_size

        while True:
            num_threads = 1
            for t in block:
                num_threads *= t
            required_registers_per_mt = num_threads * required_registers_per_thread
            if required_registers_per_mt <= available_registers_per_mp:
                return block
            else:
                largest_grid_entry_idx = max(range(len(block)), key=lambda e: block[e])
                assert block[largest_grid_entry_idx] >= 2
                block[largest_grid_entry_idx] //= 2

    @staticmethod
    def permute_block_size_according_to_layout(block_size, layout):
        """Returns modified block_size such that the fastest coordinate gets the biggest block dimension"""
        sorted_block_size = list(sorted(block_size, reverse=True))
        while len(sorted_block_size) > len(layout):
            sorted_block_size[0] *= sorted_block_size[-1]
            sorted_block_size = sorted_block_size[:-1]

        result = list(block_size)
        for l, bs in zip(reversed(layout), sorted_block_size):
            result[l] = bs
        return tuple(result[:len(layout)])


class LineIndexing(AbstractIndexing):
    """
    Indexing scheme that assigns the innermost 'line' i.e. the elements which are adjacent in memory to a 1D CUDA block.
    The fastest coordinate is indexed with thread_idx.x, the remaining coordinates are mapped to block_idx.{x,y,z}
    This indexing scheme supports up to 4 spatial dimensions, where the innermost dimensions is not larger than the
    maximum amount of threads allowed in a CUDA block (which depends on device).
    """

    def __init__(self, field, iteration_slice=None):
        available_indices = [THREAD_IDX[0]] + BLOCK_IDX
        if field.spatial_dimensions > 4:
            raise NotImplementedError("This indexing scheme supports at most 4 spatial dimensions")

        coordinates = available_indices[:field.spatial_dimensions]

        fastest_coordinate = field.layout[-1]
        coordinates[0], coordinates[fastest_coordinate] = coordinates[fastest_coordinate], coordinates[0]

        self._coordinates = coordinates
        self._iterationSlice = normalize_slice(iteration_slice, field.spatial_shape)
        self._symbolicShape = [e if isinstance(e, sp.Basic) else None for e in field.spatial_shape]

    @property
    def coordinates(self):
        return [i + offset for i, offset in zip(self._coordinates, _get_start_from_slice(self._iterationSlice))]

    def call_parameters(self, arr_shape):
        substitution_dict = {sym: value for sym, value in zip(self._symbolicShape, arr_shape) if sym is not None}

        widths = [end - start for start, end in zip(_get_start_from_slice(self._iterationSlice),
                                                    _get_end_from_slice(self._iterationSlice, arr_shape))]
        widths = sp.Matrix(widths).subs(substitution_dict)

        def get_shape_of_cuda_idx(cuda_idx):
            if cuda_idx not in self._coordinates:
                return 1
            else:
                idx = self._coordinates.index(cuda_idx)
                return widths[idx]

        return {'block': tuple([get_shape_of_cuda_idx(idx) for idx in THREAD_IDX]),
                'grid': tuple([get_shape_of_cuda_idx(idx) for idx in BLOCK_IDX])}

    def guard(self, kernel_content, arr_shape):
        return kernel_content


# -------------------------------------- Helper functions --------------------------------------------------------------

def _get_start_from_slice(iteration_slice):
    res = []
    for slice_component in iteration_slice:
        if type(slice_component) is slice:
            res.append(slice_component.start if slice_component.start is not None else 0)
        else:
            assert isinstance(slice_component, int)
            res.append(slice_component)
    return res


def _get_end_from_slice(iteration_slice, arr_shape):
    iter_slice = normalize_slice(iteration_slice, arr_shape)
    res = []
    for slice_component in iter_slice:
        if type(slice_component) is slice:
            res.append(slice_component.stop)
        else:
            assert isinstance(slice_component, int)
            res.append(slice_component + 1)
    return res


def indexing_creator_from_params(gpu_indexing, gpu_indexing_params):
    if isinstance(gpu_indexing, str):
        if gpu_indexing == 'block':
            indexing_creator = BlockIndexing
        elif gpu_indexing == 'line':
            indexing_creator = LineIndexing
        else:
            raise ValueError("Unknown GPU indexing %s. Valid values are 'block' and 'line'" % (gpu_indexing,))
        if gpu_indexing_params:
            indexing_creator = partial(indexing_creator, **gpu_indexing_params)
        return indexing_creator
    else:
        return gpu_indexing
