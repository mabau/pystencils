import numpy as np
from pystencils import Field, makeSlice
from pystencils.datahandling import DataHandling
from pystencils.parallel.blockiteration import slicedBlockIteration
from pystencils.utils import DotDict
import waLBerla as wlb


class ParallelDataHandling(DataHandling):
    GPU_DATA_PREFIX = "gpu_"

    def __init__(self, blocks, defaultGhostLayers=1, defaultLayout='SoA', dim=3):
        """
        Creates data handling based on waLBerla block storage

        :param blocks: waLBerla block storage
        :param defaultGhostLayers: nr of ghost layers used if not specified in add() method
        :param defaultLayout: layout used if no layout is given to add() method
        :param dim: dimension of scenario,
                    waLBerla always uses three dimensions, so if dim=2 the extend of the
                    z coordinate of blocks has to be 1
        """
        super(ParallelDataHandling, self).__init__()
        assert dim in (2, 3)
        self.blocks = blocks
        self.defaultGhostLayers = defaultGhostLayers
        self.defaultLayout = defaultLayout
        self._fields = DotDict()  # maps name to symbolic pystencils field
        self.dataNames = set()
        self._dim = dim
        self._fieldInformation = {}
        self._cpuGpuPairs = []
        self._customDataTransferFunctions = {}
        if self._dim == 2:
            assert self.blocks.getDomainCellBB().size[2] == 1

    @property
    def dim(self):
        return self._dim

    @property
    def fields(self):
        return self._fields

    def addCustomData(self, name, cpuCreationFunction,
                      gpuCreationFunction=None, cpuToGpuTransferFunc=None, gpuToCpuTransferFunc=None):
        self.blocks.addBlockData(name, cpuCreationFunction)
        if gpuCreationFunction:
            self.blocks.addBlockData(self.GPU_DATA_PREFIX + name, gpuCreationFunction)
            if cpuToGpuTransferFunc is None or gpuToCpuTransferFunc is None:
                raise ValueError("For GPU data, both transfer functions have to be specified")
            self._customDataTransferFunctions[name] = (cpuToGpuTransferFunc, gpuToCpuTransferFunc)

    def addArray(self, name, fSize=1, dtype=np.float64, latexName=None, ghostLayers=None, layout=None, cpu=True, gpu=False):
        if ghostLayers is None:
            ghostLayers = self.defaultGhostLayers
        if layout is None:
            layout = self.defaultLayout
        if latexName is None:
            latexName = name
        if len(self.blocks) == 0:
            raise ValueError("Data handling expects that each process has at least one block")
        if hasattr(dtype, 'type'):
            dtype = dtype.type
        if name in self.blocks[0] or self.GPU_DATA_PREFIX + name in self.blocks[0]:
            raise ValueError("Data with this name has already been added")

        self._fieldInformation[name] = {'ghostLayers': ghostLayers,
                                        'fSize': fSize,
                                        'layout': layout,
                                        'dtype': dtype}

        layoutMap = {'fzyx': wlb.field.Layout.fzyx, 'zyxf': wlb.field.Layout.zyxf,
                     'SoA': wlb.field.Layout.fzyx,  'AoS': wlb.field.Layout.zyxf}

        if cpu:
            wlb.field.addToStorage(self.blocks, name, dtype, fSize=fSize, layout=layoutMap[layout],
                                   ghostLayers=ghostLayers)
        if gpu:
            wlb.cuda.addGpuFieldToStorage(self.blocks, self.GPU_DATA_PREFIX+name, dtype, fSize=fSize,
                                          usePitchedMem=False, ghostLayers=ghostLayers, layout=layoutMap[layout])

        if cpu and gpu:
            self._cpuGpuPairs.append((name, self.GPU_DATA_PREFIX + name))

        blockBB = self.blocks.getBlockCellBB(self.blocks[0])
        shape = tuple(s + 2 * ghostLayers for s in blockBB.size)
        indexDimensions = 1 if fSize > 1 else 0
        if indexDimensions == 1:
            shape += (fSize, )

        assert all(f.name != latexName for f in self.fields.values()), "Symbolic field with this name already exists"
        self.fields[name] = Field.createFixedSize(latexName, shape, indexDimensions, dtype, layout)

    def hasData(self, name):
        return name in self._fields

    def addArrayLike(self, name, nameOfTemplateField, latexName=None, cpu=True, gpu=False):
        self.addArray(name, latexName=latexName, cpu=cpu, gpu=gpu, **self._fieldInformation[nameOfTemplateField])

    def swap(self, name1, name2, gpu=False):
        if gpu:
            name1 = self.GPU_DATA_PREFIX + name1
            name2 = self.GPU_DATA_PREFIX + name2
        for block in self.blocks:
            block[name1].swapDataPointers(block[name2])

    def accessArray(self, name, sliceObj=None, innerGhostLayers='all', outerGhostLayers='all'):
        fieldInfo = self._fieldInformation[name]
        with self.accessWrapper(name):
            if innerGhostLayers is 'all':
                innerGhostLayers = fieldInfo['ghostLayers']
            if outerGhostLayers is 'all':
                outerGhostLayers = fieldInfo['ghostLayers']

            for iterInfo in slicedBlockIteration(self.blocks, sliceObj, innerGhostLayers, outerGhostLayers):
                arr = wlb.field.toArray(iterInfo.block[name], withGhostLayers=innerGhostLayers)[iterInfo.localSlice]
                if self.fields[name].indexDimensions == 0:
                    arr = arr[..., 0]
                if self.dim == 2:
                    arr = arr[:, :, 0]
                yield arr, iterInfo

    def accessCustomData(self, name):
        with self.accessWrapper(name):
            for block in self.blocks:
                data = block[name]
                cellBB = self.blocks.getBlockCellBB(block)
                min = cellBB.min[:self.dim]
                max = tuple(e + 1 for e in cellBB.max[:self.dim])
                yield data, (min, max)

    def gatherArray(self, name, sliceObj=None, allGather=False):
        with self.accessWrapper(name):
            if sliceObj is None:
                sliceObj = makeSlice[:, :, :]
            for array in wlb.field.gatherGenerator(self.blocks, name, sliceObj, allGather):
                if self.fields[name].indexDimensions == 0:
                    array = array[..., 0]
                if self.dim == 2:
                    array = array[:, :, 0]
                yield array

    def runKernel(self, kernelFunc, *args, **kwargs):
        fieldArguments = [p.fieldName for p in kernelFunc.ast.parameters if p.isFieldPtrArgument]
        for block in self.blocks:
            fieldArgs = {fieldName: wlb.field.toArray(block[fieldName], withGhostLayers=True)
                         for fieldName in fieldArguments}
            fieldArgs.update(kwargs)
            kernelFunc(*args, **kwargs)

    def toCpu(self, name):
        if name in self._customDataTransferFunctions:
            transferFunc = self._customDataTransferFunctions[name][1]
            for block in self.blocks:
                transferFunc(block[self.GPU_DATA_PREFIX + name], block[name])
        else:
            wlb.cuda.copyFieldToCpu(self.blocks, self.GPU_DATA_PREFIX + name, name)

    def toGpu(self, name):
        if name in self._customDataTransferFunctions:
            transferFunc = self._customDataTransferFunctions[name][0]
            for block in self.blocks:
                transferFunc(block[self.GPU_DATA_PREFIX + name], block[name])
        else:
            wlb.cuda.copyFieldToGpu(self.blocks, self.GPU_DATA_PREFIX + name, name)

    def allToCpu(self):
        for cpuName, gpuName in self._cpuGpuPairs:
            wlb.cuda.copyFieldToCpu(self.blocks, gpuName, cpuName)
        for name in self._customDataTransferFunctions.keys():
            self.toCpu(name)

    def allToGpu(self):
        for cpuName, gpuName in self._cpuGpuPairs:
            wlb.cuda.copyFieldToGpu(self.blocks, gpuName, cpuName)
        for name in self._customDataTransferFunctions.keys():
            self.toGpu(name)

    def synchronizationFunctionCPU(self, names, stencil=None, buffered=True, **kwargs):
        return self._synchronizationFunction(names, stencil, buffered, 'cpu')

    def synchronizationFunctionGPU(self, names, stencil=None, buffered=True, **kwargs):
        return self._synchronizationFunction(names, stencil, buffered, 'gpu')

    def _synchronizationFunction(self, names, stencil, buffered, target):
        if stencil is None:
            stencil = 'D3Q27' if self.dim == 3 else 'D2Q9'

        if not hasattr(names, '__len__') or type(names) is str:
            names = [names]

        createScheme = wlb.createUniformBufferedScheme if buffered else wlb.createUniformDirectScheme
        if target == 'cpu':
            createPacking = wlb.field.createPackInfo if buffered else wlb.field.createMPIDatatypeInfo
        elif target == 'gpu':
            createPacking = wlb.cuda.createPackInfo if buffered else wlb.cuda.createMPIDatatypeInfo
            names = [self.GPU_DATA_PREFIX + name for name in names]

        syncFunction = createScheme(self.blocks, stencil)
        for name in names:
            syncFunction.addDataToCommunicate(createPacking(self.blocks, name))

        return syncFunction
