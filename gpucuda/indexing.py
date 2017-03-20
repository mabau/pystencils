import sympy as sp
import math
import pycuda.driver as cuda
import pycuda.autoinit

from pystencils.astnodes import Conditional, Block

BLOCK_IDX = list(sp.symbols("blockIdx.x blockIdx.y blockIdx.z"))
THREAD_IDX = list(sp.symbols("threadIdx.x threadIdx.y threadIdx.z"))

# Part 1:
#  given a field and the number of ghost layers, return the x, y and z coordinates
#  dependent on CUDA thread and block indices

# Part 2:
#  given the actual field size, determine the call parameters i.e. # of blocks and threads


class LineIndexing:
    def __init__(self, field, ghostLayers):
        availableIndices = [THREAD_IDX[0]] + BLOCK_IDX
        if field.spatialDimensions > 4:
            raise NotImplementedError("This indexing scheme supports at most 4 spatial dimensions")

        coordinates = availableIndices[:field.spatialDimensions]

        fastestCoordinate = field.layout[-1]
        coordinates[0], coordinates[fastestCoordinate] = coordinates[fastestCoordinate], coordinates[0]

        self._coordiantesNoGhostLayer = coordinates
        self._coordinates = [i + ghostLayers for i in coordinates]
        self._ghostLayers = ghostLayers

    @property
    def coordinates(self):
        return self._coordinates

    def getCallParameters(self, arrShape):
        def getShapeOfCudaIdx(cudaIdx):
            if cudaIdx not in self._coordiantesNoGhostLayer:
                return 1
            else:
                return arrShape[self._coordiantesNoGhostLayer.index(cudaIdx)] - 2 * self._ghostLayers

        return {'block': tuple([getShapeOfCudaIdx(idx) for idx in THREAD_IDX]),
                'grid': tuple([getShapeOfCudaIdx(idx) for idx in BLOCK_IDX])}

    def guard(self, kernelContent, arrShape):
        return kernelContent

    @property
    def indexVariables(self):
        return BLOCK_IDX + THREAD_IDX


class BlockIndexing:
    def __init__(self, field, ghostLayers, blockSize=(256, 8, 1), permuteBlockSizeDependentOnLayout=True):
        if field.spatialDimensions > 3:
            raise NotImplementedError("This indexing scheme supports at most 3 spatial dimensions")

        if permuteBlockSizeDependentOnLayout:
            blockSize = self.permuteBlockSizeAccordingToLayout(blockSize, field.layout)

        self._blockSize = self.limitBlockSizeToDeviceMaximum(blockSize)
        self._coordinates = [blockIndex * bs + threadIndex + ghostLayers
                             for blockIndex, bs, threadIndex in zip(BLOCK_IDX, blockSize, THREAD_IDX)]

        self._coordinates = self._coordinates[:field.spatialDimensions]
        self._ghostLayers = ghostLayers

    @staticmethod
    def limitBlockSizeToDeviceMaximum(blockSize):
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
    def permuteBlockSizeAccordingToLayout(blockSize, layout):
        """The fastest coordinate gets the biggest block dimension"""
        sortedBlockSize = list(sorted(blockSize, reverse=True))
        while len(sortedBlockSize) > len(layout):
            sortedBlockSize[0] *= sortedBlockSize[-1]
            sortedBlockSize = sortedBlockSize[:-1]

        result = list(blockSize)
        for l, bs in zip(reversed(layout), sortedBlockSize):
            result[l] = bs
        return tuple(result[:len(layout)])

    @property
    def coordinates(self):
        return self._coordinates

    def getCallParameters(self, arrShape):
        dim = len(self._coordinates)
        arrShape = arrShape[:dim]
        grid = tuple(math.ceil(length / blockSize) for length, blockSize in zip(arrShape, self._blockSize))
        extendBs = (1,) * (3 - len(self._blockSize))
        extendGr = (1,) * (3 - len(grid))
        return {'block': self._blockSize + extendBs,
                'grid': grid + extendGr}

    def guard(self, kernelContent, arrShape):
        dim = len(self._coordinates)
        arrShape = arrShape[:dim]
        conditions = [c < shapeComponent - self._ghostLayers
                      for c, shapeComponent in zip(self._coordinates, arrShape)]
        condition = conditions[0]
        for c in conditions[1:]:
            condition = sp.And(condition, c)
        return Block([Conditional(condition, kernelContent)])

    @property
    def indexVariables(self):
        return BLOCK_IDX + THREAD_IDX

if __name__ == '__main__':
    bs = BlockIndexing.permuteBlockSizeAccordingToLayout((256, 8, 1), (0,))
    bs = BlockIndexing.limitBlockSizeToDeviceMaximum(bs)
    print(bs)
