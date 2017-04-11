import abc

import sympy as sp
import math
import pycuda.driver as cuda
import pycuda.autoinit

from pystencils.astnodes import Conditional, Block
from pystencils.slicing import normalizeSlice

BLOCK_IDX = list(sp.symbols("blockIdx.x blockIdx.y blockIdx.z"))
THREAD_IDX = list(sp.symbols("threadIdx.x threadIdx.y threadIdx.z"))


class AbstractIndexing(abc.ABCMeta('ABC', (object,), {})):
    """
    Abstract base class for all Indexing classes. An Indexing class defines how a multidimensional
    field is mapped to CUDA's block and grid system. It calculates indices based on CUDA's thread and block indices
    and computes the number of blocks and threads a kernel is started with. The Indexing class is created with
    a pystencils field, a slice to iterate over, and further optional parameters that must have default values.
    """

    @abc.abstractproperty
    def coordinates(self):
        """Returns a sequence of coordinate expressions for (x,y,z) depending on symbolic CUDA block and thread indices.
        These symbolic indices can be obtained with the method `indexVariables` """

    @property
    def indexVariables(self):
        """Sympy symbols for CUDA's block and thread indices"""
        return BLOCK_IDX + THREAD_IDX

    @abc.abstractmethod
    def getCallParameters(self, arrShape, functionToCall):
        """
        Determine grid and block size for kernel call
        :param arrShape: the numeric (not symbolic) shape of the array
        :param functionToCall: compile kernel function that should be called. Use this object to get information
                               about required resources like number of registers
        :return: dict with keys 'blocks' and 'threads' with tuple values for number of (x,y,z) threads and blocks
                 the kernel should be started with
        """

    @abc.abstractmethod
    def guard(self, kernelContent, arrShape):
        """
        In some indexing schemes not all threads of a block execute the kernel content.
        This function can return a Conditional ast node, defining this execution guard.
        :param kernelContent: the actual kernel contents which can e.g. be put into the Conditional node as true block
        :param arrShape: the numeric or symbolic shape of the field
        :return: ast node, which is put inside the kernel function
        """


# -------------------------------------------- Implementations ---------------------------------------------------------


class BlockIndexing(AbstractIndexing):
    """Generic indexing scheme that maps sub-blocks of an array to CUDA blocks."""

    def __init__(self, field, iterationSlice=None,
                 blockSize=(256, 8, 1), permuteBlockSizeDependentOnLayout=True):
        """
        Creates
        :param field: pystencils field (common to all Indexing classes)
        :param iterationSlice: slice that defines rectangular subarea which is iterated over
        :param permuteBlockSizeDependentOnLayout: if True the blockSize is permuted such that the fastest coordinate
                                                  gets the largest amount of threads
        """
        if field.spatialDimensions > 3:
            raise NotImplementedError("This indexing scheme supports at most 3 spatial dimensions")

        if permuteBlockSizeDependentOnLayout:
            blockSize = self.permuteBlockSizeAccordingToLayout(blockSize, field.layout)

        blockSize = self.limitBlockSizeToDeviceMaximum(blockSize)
        self._blockSize = blockSize
        self._iterationSlice = normalizeSlice(iterationSlice, field.spatialShape)
        self._dim = field.spatialDimensions
        self._symbolicShape = [e if isinstance(e, sp.Basic) else None for e in field.spatialShape]

    @property
    def coordinates(self):
        offsets = _getStartFromSlice(self._iterationSlice)
        coordinates = [blockIndex * bs + threadIdx + off
                       for blockIndex, bs, threadIdx, off in zip(BLOCK_IDX, self._blockSize, THREAD_IDX, offsets)]

        return coordinates[:self._dim]

    def getCallParameters(self, arrShape, functionToCall):
        substitutionDict = {sym: value for sym, value in zip(self._symbolicShape, arrShape) if sym is not None}

        widths = [end - start for start, end in zip(_getStartFromSlice(self._iterationSlice),
                                                    _getEndFromSlice(self._iterationSlice, arrShape))]
        widths = sp.Matrix(widths).subs(substitutionDict)

        grid = tuple(math.ceil(length / blockSize) for length, blockSize in zip(widths, self._blockSize))
        extendBs = (1,) * (3 - len(self._blockSize))
        extendGr = (1,) * (3 - len(grid))

        return {'block': self._blockSize + extendBs,
                'grid': grid + extendGr}

    def guard(self, kernelContent, arrShape):
        arrShape = arrShape[:self._dim]
        conditions = [c < end
                      for c, end in zip(self.coordinates, _getEndFromSlice(self._iterationSlice, arrShape))]
        condition = conditions[0]
        for c in conditions[1:]:
            condition = sp.And(condition, c)
        return Block([Conditional(condition, kernelContent)])

    @staticmethod
    def limitBlockSizeToDeviceMaximum(blockSize):
        """
        Changes blocksize according to match device limits according to the following rules:
        1) if the total amount of threads is too big for the current device, the biggest coordinate is divided by 2.
        2) next, if one component is still too big, the component which is too big is divided by 2 and the smallest
           component is multiplied by 2, such that the total amount of threads stays the same
        Returns the altered blockSize
        """
        # Get device limits
        da = cuda.device_attribute
        device = cuda.Context.get_device()

        blockSize = list(blockSize)
        maxThreads = device.get_attribute(da.MAX_THREADS_PER_BLOCK)
        maxBlockSize = [device.get_attribute(a)
                        for a in (da.MAX_BLOCK_DIM_X, da.MAX_BLOCK_DIM_Y, da.MAX_BLOCK_DIM_Z)]

        def prod(seq):
            result = 1
            for e in seq:
                result *= e
            return result

        def getIndexOfTooBigElement(blockSize):
            for i, bs in enumerate(blockSize):
                if bs > maxBlockSize[i]:
                    return i
            return None

        def getIndexOfTooSmallElement(blockSize):
            for i, bs in enumerate(blockSize):
                if bs // 2 <= maxBlockSize[i]:
                    return i
            return None

        # Reduce the total number of threads if necessary
        while prod(blockSize) > maxThreads:
            itemToReduce = blockSize.index(max(blockSize))
            for i, bs in enumerate(blockSize):
                if bs > maxBlockSize[i]:
                    itemToReduce = i
            blockSize[itemToReduce] //= 2

        # Cap individual elements
        tooBigElementIndex = getIndexOfTooBigElement(blockSize)
        while tooBigElementIndex is not None:
            tooSmallElementIndex = getIndexOfTooSmallElement(blockSize)
            blockSize[tooSmallElementIndex] *= 2
            blockSize[tooBigElementIndex] //= 2
            tooBigElementIndex = getIndexOfTooBigElement(blockSize)

        return tuple(blockSize)

    @staticmethod
    def limitBlockSizeByRegisterRestriction(blockSize, requiredRegistersPerThread, device=None):
        """Shrinks the blockSize if there are too many registers used per multiprocessor.
        This is not done automatically, since the requiredRegistersPerThread are not known before compilation.
        They can be obtained by ``func.num_regs`` from a pycuda function.
        :returns smaller blockSize if too many registers are used.
        """
        da = cuda.device_attribute
        if device is None:
            device = cuda.Context.get_device()
        availableRegistersPerMP = device.get_attribute(da.MAX_REGISTERS_PER_MULTIPROCESSOR)

        block = blockSize

        while True:
            numThreads = 1
            for t in block:
                numThreads *= t
            requiredRegistersPerMT = numThreads * requiredRegistersPerThread
            if requiredRegistersPerMT <= availableRegistersPerMP:
                return block
            else:
                largestGridEntryIdx = max(range(len(block)), key=lambda e: block[e])
                assert block[largestGridEntryIdx] >= 2
                block[largestGridEntryIdx] //= 2

    @staticmethod
    def permuteBlockSizeAccordingToLayout(blockSize, layout):
        """Returns modified blockSize such that the fastest coordinate gets the biggest block dimension"""
        sortedBlockSize = list(sorted(blockSize, reverse=True))
        while len(sortedBlockSize) > len(layout):
            sortedBlockSize[0] *= sortedBlockSize[-1]
            sortedBlockSize = sortedBlockSize[:-1]

        result = list(blockSize)
        for l, bs in zip(reversed(layout), sortedBlockSize):
            result[l] = bs
        return tuple(result[:len(layout)])


class LineIndexing(AbstractIndexing):
    """
    Indexing scheme that assigns the innermost 'line' i.e. the elements which are adjacent in memory to a 1D CUDA block.
    The fastest coordinate is indexed with threadIdx.x, the remaining coordinates are mapped to blockIdx.{x,y,z}
    This indexing scheme supports up to 4 spatial dimensions, where the innermost dimensions is not larger than the
    maximum amount of threads allowed in a CUDA block (which depends on device).
    """

    def __init__(self, field, iterationSlice=None):
        availableIndices = [THREAD_IDX[0]] + BLOCK_IDX
        if field.spatialDimensions > 4:
            raise NotImplementedError("This indexing scheme supports at most 4 spatial dimensions")

        coordinates = availableIndices[:field.spatialDimensions]

        fastestCoordinate = field.layout[-1]
        coordinates[0], coordinates[fastestCoordinate] = coordinates[fastestCoordinate], coordinates[0]

        self._coordinates = coordinates
        self._iterationSlice = normalizeSlice(iterationSlice, field.spatialShape)
        self._symbolicShape = [e if isinstance(e, sp.Basic) else None for e in field.spatialShape]

    @property
    def coordinates(self):
        return [i + offset for i, offset in zip(self._coordinates, _getStartFromSlice(self._iterationSlice))]

    def getCallParameters(self, arrShape, functionToCall):
        substitutionDict = {sym: value for sym, value in zip(self._symbolicShape, arrShape) if sym is not None}

        widths = [end - start for start, end in zip(_getStartFromSlice(self._iterationSlice),
                                                    _getEndFromSlice(self._iterationSlice, arrShape))]
        widths = sp.Matrix(widths).subs(substitutionDict)

        def getShapeOfCudaIdx(cudaIdx):
            if cudaIdx not in self._coordinates:
                return 1
            else:
                idx = self._coordinates.index(cudaIdx)
                return int(widths[idx])

        return {'block': tuple([getShapeOfCudaIdx(idx) for idx in THREAD_IDX]),
                'grid': tuple([getShapeOfCudaIdx(idx) for idx in BLOCK_IDX])}

    def guard(self, kernelContent, arrShape):
        return kernelContent


# -------------------------------------- Helper functions --------------------------------------------------------------

def _getStartFromSlice(iterationSlice):
    res = []
    for sliceComponent in iterationSlice:
        if type(sliceComponent) is slice:
            res.append(sliceComponent.start if sliceComponent.start is not None else 0)
        else:
            assert isinstance(sliceComponent, int)
            res.append(sliceComponent)
    return res


def _getEndFromSlice(iterationSlice, arrShape):
    iterSlice = normalizeSlice(iterationSlice, arrShape)
    res = []
    for sliceComponent in iterSlice:
        if type(sliceComponent) is slice:
            res.append(sliceComponent.stop)
        else:
            assert isinstance(sliceComponent, int)
            res.append(sliceComponent + 1)
    return res

