import abc
from functools import partial
import math
from typing import List, Tuple

import sympy as sp
from sympy.core.cache import cacheit

from pystencils.astnodes import Block, Conditional, SympyAssignment
from pystencils.typing import TypedSymbol, create_type
from pystencils.integer_functions import div_ceil, div_floor
from pystencils.sympyextensions import is_integer_sequence, prod


class ThreadIndexingSymbol(TypedSymbol):
    def __new__(cls, *args, **kwds):
        obj = ThreadIndexingSymbol.__xnew_cached_(cls, *args, **kwds)
        return obj

    def __new_stage2__(cls, name, dtype, *args, **kwargs):
        obj = super(ThreadIndexingSymbol, cls).__xnew__(cls, name, dtype, *args, **kwargs)
        return obj

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))


BLOCK_IDX = [ThreadIndexingSymbol("blockIdx." + coord, create_type("int32")) for coord in ('x', 'y', 'z')]
THREAD_IDX = [ThreadIndexingSymbol("threadIdx." + coord, create_type("int32")) for coord in ('x', 'y', 'z')]
BLOCK_DIM = [ThreadIndexingSymbol("blockDim." + coord, create_type("int32")) for coord in ('x', 'y', 'z')]
GRID_DIM = [ThreadIndexingSymbol("gridDim." + coord, create_type("int32")) for coord in ('x', 'y', 'z')]


class AbstractIndexing(abc.ABC):
    """
    Abstract base class for all Indexing classes. An Indexing class defines how an iteration space is mapped
    to GPU's block and grid system. It calculates indices based on GPU's thread and block indices
    and computes the number of blocks and threads a kernel is started with.
    The Indexing class is created with an iteration space that is given as list of slices to determine start, stop
    and the step size for each coordinate. Further the data_layout is given as tuple to determine the fast and slow
    coordinates. This is important to get an optimal mapping of coordinates to GPU threads.
    """

    def __init__(self, iteration_space: Tuple[slice], data_layout: Tuple):
        for iter_space in iteration_space:
            assert isinstance(iter_space, slice), f"iteration_space must be of type Tuple[slice], " \
                                                  f"not tuple of type {type(iter_space)}"
        self._iteration_space = iteration_space
        self._data_layout = data_layout
        self._dim = len(iteration_space)

    @property
    def iteration_space(self):
        """Iteration space to loop over"""
        return self._iteration_space

    @property
    def data_layout(self):
        """Data layout of the kernels arrays"""
        return self._data_layout

    @property
    def dim(self):
        """Number of spatial dimensions"""
        return self._dim

    @property
    @abc.abstractmethod
    def coordinates(self):
        """Returns a sequence of coordinate expressions for (x,y,z) depending on symbolic GPU block and thread indices.
        These symbolic indices can be obtained with the method `index_variables` """

    @property
    def index_variables(self):
        """Sympy symbols for GPU's block and thread indices, and block and grid dimensions. """
        return BLOCK_IDX + THREAD_IDX + BLOCK_DIM + GRID_DIM

    @abc.abstractmethod
    def get_loop_ctr_assignments(self, loop_counter_symbols) -> List[SympyAssignment]:
        """Adds assignments for the loop counter symbols depending on the gpu threads.

        Args:
            loop_counter_symbols: typed symbols representing the loop counters
        Returns:
            assignments for the loop counters
        """

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

    @abc.abstractmethod
    def max_threads_per_block(self):
        """Return maximal number of threads per block for launch bounds. If this cannot be determined without
        knowing the array shape return None for unknown """

    @abc.abstractmethod
    def symbolic_parameters(self):
        """Set of symbols required in call_parameters code"""


# -------------------------------------------- Implementations ---------------------------------------------------------


class BlockIndexing(AbstractIndexing):
    """Generic indexing scheme that maps sub-blocks of an array to GPU blocks.

    Args:
        iteration_space: list of slices to determine start, stop and the step size for each coordinate
        data_layout: tuple specifying loop order with innermost loop last.
                     This is the same format as returned by `Field.layout`.
        permute_block_size_dependent_on_layout: if True the block_size is permuted such that the fastest coordinate
                                                gets the largest amount of threads
        compile_time_block_size: compile in concrete block size, otherwise the gpu variable 'blockDim' is used
        maximum_block_size: maximum block size that is possible for the GPU. Set to 'auto' to let cupy define the
                            maximum block size from the device properties
        device_number: device number of the used GPU. By default, the zeroth device is used.
    """

    def __init__(self, iteration_space: Tuple[slice], data_layout: Tuple[int],
                 block_size=(128, 2, 1), permute_block_size_dependent_on_layout=True, compile_time_block_size=False,
                 maximum_block_size=(1024, 1024, 64), device_number=None):
        super(BlockIndexing, self).__init__(iteration_space, data_layout)

        if self._dim > 4:
            raise NotImplementedError("This indexing scheme supports at most 4 spatial dimensions")

        if permute_block_size_dependent_on_layout and self._dim < 4:
            block_size = self.permute_block_size_according_to_layout(block_size, data_layout)

        self._block_size = block_size
        if maximum_block_size == 'auto':
            assert device_number is not None, 'If "maximum_block_size" is set to "auto" a device number must be stated'
            # Get device limits
            import cupy as cp
            # See https://github.com/cupy/cupy/issues/7676
            if cp.cuda.runtime.is_hip:
                maximum_block_size = tuple(cp.cuda.runtime.deviceGetAttribute(i, device_number) for i in range(26, 29))
            else:
                da = cp.cuda.Device(device_number).attributes
                maximum_block_size = tuple(da[f"MaxBlockDim{c}"] for c in ["X", "Y", "Z"])

        self._maximum_block_size = maximum_block_size
        self._compile_time_block_size = compile_time_block_size
        self._device_number = device_number

    @property
    def cuda_indices(self):
        block_size = self._block_size if self._compile_time_block_size else BLOCK_DIM
        indices = [block_index * bs + thread_idx
                   for block_index, bs, thread_idx in zip(BLOCK_IDX, block_size, THREAD_IDX)]

        return indices[:self._dim]

    @property
    def coordinates(self):
        if self._dim < 4:
            coordinates = [c + iter_slice.start for c, iter_slice in zip(self.cuda_indices, self._iteration_space)]
            return coordinates[:self._dim]
        else:
            coordinates = list()
            width = self._iteration_space[1].stop - self.iteration_space[1].start
            coordinates.append(div_floor(self.cuda_indices[0], width))
            coordinates.append(sp.Mod(self.cuda_indices[0], width))
            coordinates.append(self.cuda_indices[1] + self.iteration_space[2].start)
            coordinates.append(self.cuda_indices[2] + self.iteration_space[3].start)
            return coordinates

    def get_loop_ctr_assignments(self, loop_counter_symbols):
        return _loop_ctr_assignments(loop_counter_symbols, self.coordinates, self._iteration_space)

    def call_parameters(self, arr_shape):
        numeric_iteration_slice = _get_numeric_iteration_slice(self._iteration_space, arr_shape)
        widths = _get_widths(numeric_iteration_slice)

        if len(widths) > 3:
            widths = [widths[0] * widths[1], widths[2], widths[3]]

        extend_bs = (1,) * (3 - len(self._block_size))
        block_size = self._block_size + extend_bs
        if not self._compile_time_block_size:
            assert len(block_size) == 3
            adapted_block_size = []
            for i in range(len(widths)):
                factor = div_floor(prod(block_size[:i]), prod(adapted_block_size))
                adapted_block_size.append(sp.Min(block_size[i] * factor, widths[i]))
            extend_adapted_bs = (1,) * (3 - len(adapted_block_size))
            block_size = tuple(adapted_block_size) + extend_adapted_bs

        block_size = tuple(sp.Min(bs, max_bs) for bs, max_bs in zip(block_size, self._maximum_block_size))
        grid = tuple(div_ceil(length, block_size) for length, block_size in zip(widths, block_size))
        extend_gr = (1,) * (3 - len(grid))

        return {'block': block_size,
                'grid': grid + extend_gr}

    def guard(self, kernel_content, arr_shape):
        arr_shape = arr_shape[:self._dim]
        if len(self._iteration_space) - 1 == len(arr_shape):
            numeric_iteration_slice = _get_numeric_iteration_slice(self._iteration_space[1:], arr_shape)
            numeric_iteration_slice = [self.iteration_space[0]] + numeric_iteration_slice
        else:
            assert len(self._iteration_space) == len(arr_shape), "Iteration space must be equal to the array shape"
            numeric_iteration_slice = _get_numeric_iteration_slice(self._iteration_space, arr_shape)
        end = [s.stop if s.stop != 0 else 1 for s in numeric_iteration_slice]

        if self._dim < 4:
            conditions = [c < e for c, e in zip(self.coordinates, end)]
        else:
            end = [end[0] * end[1], end[2], end[3]]
            coordinates = [c + iter_slice.start for c, iter_slice in zip(self.cuda_indices, self._iteration_space[1:])]
            conditions = [c < e for c, e in zip(coordinates, end)]
        condition = conditions[0]
        for c in conditions[1:]:
            condition = sp.And(condition, c)
        return Block([Conditional(condition, kernel_content)])

    def numeric_iteration_space(self, arr_shape):
        return _get_numeric_iteration_slice(self._iteration_space, arr_shape)

    def limit_block_size_by_register_restriction(self, block_size, required_registers_per_thread):
        """Shrinks the block_size if there are too many registers used per block.
        This is not done automatically, since the required_registers_per_thread are not known before compilation.
        They can be obtained by ``func.num_regs`` from a cupy function.
        Args:
            block_size: used block size that is target for limiting
            required_registers_per_thread: needed registers per thread
        returns: smaller block_size if too many registers are used.
        """
        import cupy as cp

        # See https://github.com/cupy/cupy/issues/7676
        if cp.cuda.runtime.is_hip:
            max_registers_per_block = cp.cuda.runtime.deviceGetAttribute(71, self._device_number)
        else:
            device = cp.cuda.Device(self._device_number)
            da = device.attributes
            max_registers_per_block = da.get("MaxRegistersPerBlock")

        result = list(block_size)
        while True:
            required_registers = math.prod(result) * required_registers_per_thread
            if required_registers <= max_registers_per_block:
                return result
            else:
                largest_list_entry_idx = max(range(len(result)), key=lambda e: result[e])
                assert result[largest_list_entry_idx] >= 2
                result[largest_list_entry_idx] //= 2

    @staticmethod
    def permute_block_size_according_to_layout(block_size, layout):
        """Returns modified block_size such that the fastest coordinate gets the biggest block dimension"""
        if not is_integer_sequence(block_size):
            return block_size
        sorted_block_size = list(sorted(block_size, reverse=True))
        while len(sorted_block_size) > len(layout):
            sorted_block_size[0] *= sorted_block_size[-1]
            sorted_block_size = sorted_block_size[:-1]

        result = list(block_size)
        for l, bs in zip(reversed(layout), sorted_block_size):
            result[l] = bs
        return tuple(result[:len(layout)])

    def max_threads_per_block(self):
        if is_integer_sequence(self._block_size):
            return prod(self._block_size)
        else:
            return None

    def symbolic_parameters(self):
        return set(b for b in self._block_size if isinstance(b, sp.Symbol))


class LineIndexing(AbstractIndexing):
    """
    Indexing scheme that assigns the innermost 'line' i.e. the elements which are adjacent in memory to a 1D GPU block.
    The fastest coordinate is indexed with thread_idx.x, the remaining coordinates are mapped to block_idx.{x,y,z}
    This indexing scheme supports up to 4 spatial dimensions, where the innermost dimensions is not larger than the
    maximum amount of threads allowed in a GPU block (which depends on device).

    Args:
        iteration_space: list of slices to determine start, stop and the step size for each coordinate
        data_layout: tuple to determine the fast and slow coordinates.
    """

    def __init__(self, iteration_space: Tuple[slice], data_layout: Tuple):
        super(LineIndexing, self).__init__(iteration_space, data_layout)

        if len(iteration_space) > 4:
            raise NotImplementedError("This indexing scheme supports at most 4 spatial dimensions")

    @property
    def cuda_indices(self):
        available_indices = [THREAD_IDX[0]] + BLOCK_IDX
        coordinates = available_indices[:self.dim]

        fastest_coordinate = self.data_layout[-1]
        coordinates[0], coordinates[fastest_coordinate] = coordinates[fastest_coordinate], coordinates[0]

        return coordinates

    @property
    def coordinates(self):
        return [i + o.start for i, o in zip(self.cuda_indices, self._iteration_space)]

    def get_loop_ctr_assignments(self, loop_counter_symbols):
        return _loop_ctr_assignments(loop_counter_symbols, self.coordinates, self._iteration_space)

    def call_parameters(self, arr_shape):
        numeric_iteration_slice = _get_numeric_iteration_slice(self._iteration_space, arr_shape)
        widths = _get_widths(numeric_iteration_slice)

        def get_shape_of_cuda_idx(cuda_idx):
            if cuda_idx not in self.cuda_indices:
                return 1
            else:
                idx = self.cuda_indices.index(cuda_idx)
                return widths[idx]

        return {'block': tuple([get_shape_of_cuda_idx(idx) for idx in THREAD_IDX]),
                'grid': tuple([get_shape_of_cuda_idx(idx) for idx in BLOCK_IDX])}

    def guard(self, kernel_content, arr_shape):
        return kernel_content

    def max_threads_per_block(self):
        return None

    def symbolic_parameters(self):
        return set()

    def numeric_iteration_space(self, arr_shape):
        return _get_numeric_iteration_slice(self._iteration_space, arr_shape)


# -------------------------------------- Helper functions --------------------------------------------------------------

def _get_numeric_iteration_slice(iteration_slice, arr_shape):
    res = []
    for slice_component, shape in zip(iteration_slice, arr_shape):
        result_slice = slice_component
        if not isinstance(result_slice.start, int):
            start = result_slice.start
            assert len(start.free_symbols) == 1
            start = start.subs({symbol: shape for symbol in start.free_symbols})
            result_slice = slice(start, result_slice.stop, result_slice.step)
        if not isinstance(result_slice.stop, int):
            stop = result_slice.stop
            assert len(stop.free_symbols) == 1
            stop = stop.subs({symbol: shape for symbol in stop.free_symbols})
            result_slice = slice(result_slice.start, stop, result_slice.step)
        assert isinstance(result_slice.step, int)
        res.append(result_slice)
    return res


def _get_widths(iteration_slice):
    widths = []
    for iter_slice in iteration_slice:
        step = iter_slice.step
        assert isinstance(step, int), f"Step can only be of type int not of type {type(step)}"
        start = iter_slice.start
        stop = iter_slice.stop
        if step == 1:
            if stop - start == 0:
                widths.append(1)
            else:
                widths.append(stop - start)
        else:
            width = (stop - start) / step
            if isinstance(width, int):
                widths.append(width)
            elif isinstance(width, float):
                widths.append(math.ceil(width))
            else:
                widths.append(div_ceil(stop - start, step))
    return widths


def _loop_ctr_assignments(loop_counter_symbols, coordinates, iteration_space):
    loop_ctr_assignments = []
    for loop_counter, coordinate, iter_slice in zip(loop_counter_symbols, coordinates, iteration_space):
        if isinstance(iter_slice, slice) and iter_slice.step > 1:
            offset = (iter_slice.step * iter_slice.start) - iter_slice.start
            loop_ctr_assignments.append(SympyAssignment(loop_counter, coordinate * iter_slice.step - offset))
        elif iter_slice.start == iter_slice.stop:
            loop_ctr_assignments.append(SympyAssignment(loop_counter, 0))
        else:
            loop_ctr_assignments.append(SympyAssignment(loop_counter, coordinate))

    return loop_ctr_assignments


def indexing_creator_from_params(gpu_indexing, gpu_indexing_params):
    if isinstance(gpu_indexing, str):
        if gpu_indexing == 'block':
            indexing_creator = BlockIndexing
        elif gpu_indexing == 'line':
            indexing_creator = LineIndexing
        else:
            raise ValueError(f"Unknown GPU indexing {gpu_indexing}. Valid values are 'block' and 'line'")
        if gpu_indexing_params:
            indexing_creator = partial(indexing_creator, **gpu_indexing_params)
        return indexing_creator
    else:
        return gpu_indexing
