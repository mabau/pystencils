import numpy as np
from pystencils import Field
from pystencils.datahandling.datahandling_interface import DataHandling
from pystencils.parallel.blockiteration import slicedBlockIteration, blockIteration
from pystencils.utils import DotDict
import waLBerla as wlb
import warnings


class ParallelDataHandling(DataHandling):
    GPU_DATA_PREFIX = "gpu_"
    VTK_COUNTER = 0

    def __init__(self, blocks, defaultGhostLayers=1, defaultLayout='SoA', dim=3, defaultTarget='cpu'):
        """
        Creates data handling based on waLBerla block storage

        :param blocks: waLBerla block storage
        :param defaultGhostLayers: nr of ghost layers used if not specified in add() method
        :param defaultLayout: layout used if no layout is given to add() method
        :param dim: dimension of scenario,
                    waLBerla always uses three dimensions, so if dim=2 the extend of the
                    z coordinate of blocks has to be 1
        :param defaultTarget: either 'cpu' or 'gpu' . If set to 'gpu' for each array also a GPU version is allocated
                              if not overwritten in addArray, and synchronization functions are for the GPU by default
        """
        super(ParallelDataHandling, self).__init__()
        assert dim in (2, 3)
        self.blocks = blocks
        self.defaultGhostLayers = defaultGhostLayers
        self.defaultLayout = defaultLayout
        self._fields = DotDict()  # maps name to symbolic pystencils field
        self._fieldNameToCpuDataName = {}
        self._fieldNameToGpuDataName = {}
        self.dataNames = set()
        self._dim = dim
        self._fieldInformation = {}
        self._cpuGpuPairs = []
        self._customDataTransferFunctions = {}
        self._customDataNames = []
        self._reduceMap = {
            'sum': wlb.mpi.SUM,
            'min': wlb.mpi.MIN,
            'max': wlb.mpi.MAX,
        }

        if self._dim == 2:
            assert self.blocks.getDomainCellBB().size[2] == 1
        self.defaultTarget = defaultTarget

    @property
    def dim(self):
        return self._dim

    @property
    def shape(self):
        return self.blocks.getDomainCellBB().size[:self.dim]

    @property
    def periodicity(self):
        return self.blocks.periodic[:self._dim]

    @property
    def fields(self):
        return self._fields

    def ghostLayersOfField(self, name):
        return self._fieldInformation[name]['ghostLayers']

    def fSize(self, name):
        return self._fieldInformation[name]['fSize']

    def addCustomData(self, name, cpuCreationFunction,
                      gpuCreationFunction=None, cpuToGpuTransferFunc=None, gpuToCpuTransferFunc=None):
        if cpuCreationFunction and gpuCreationFunction:
            if cpuToGpuTransferFunc is None or gpuToCpuTransferFunc is None:
                raise ValueError("For GPU data, both transfer functions have to be specified")
            self._customDataTransferFunctions[name] = (cpuToGpuTransferFunc, gpuToCpuTransferFunc)

        if cpuCreationFunction:
            self.blocks.addBlockData(name, cpuCreationFunction)
        if gpuCreationFunction:
            self.blocks.addBlockData(self.GPU_DATA_PREFIX + name, gpuCreationFunction)
        self._customDataNames.append(name)

    def addArray(self, name, fSize=1, dtype=np.float64, latexName=None, ghostLayers=None,
                 layout=None, cpu=True, gpu=None):
        if ghostLayers is None:
            ghostLayers = self.defaultGhostLayers
        if gpu is None:
            gpu = self.defaultTarget == 'gpu'
        if layout is None:
            layout = self.defaultLayout
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
                     'f': wlb.field.Layout.fzyx,
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
        shape = tuple(s + 2 * ghostLayers for s in blockBB.size[:self.dim])
        indexDimensions = 1 if fSize > 1 else 0
        if indexDimensions == 1:
            shape += (fSize, )

        assert all(f.name != name for f in self.fields.values()), "Symbolic field with this name already exists"

        self.fields[name] = Field.createGeneric(name, self.dim, dtype, indexDimensions, layout,
                                                indexShape=(fSize,) if indexDimensions > 0 else None)
        self.fields[name].latexName = latexName
        self._fieldNameToCpuDataName[name] = name
        if gpu:
            self._fieldNameToGpuDataName[name] = self.GPU_DATA_PREFIX + name
        return self.fields[name]

    def hasData(self, name):
        return name in self._fields

    @property
    def arrayNames(self):
        return tuple(self.fields.keys())

    @property
    def customDataNames(self):
        return tuple(self._customDataNames)

    def addArrayLike(self, name, nameOfTemplateField, latexName=None, cpu=True, gpu=None):
        return self.addArray(name, latexName=latexName, cpu=cpu, gpu=gpu, **self._fieldInformation[nameOfTemplateField])

    def swap(self, name1, name2, gpu=False):
        if gpu:
            name1 = self.GPU_DATA_PREFIX + name1
            name2 = self.GPU_DATA_PREFIX + name2
        for block in self.blocks:
            block[name1].swapDataPointers(block[name2])

    def iterate(self, sliceObj=None, gpu=False, ghostLayers=True, innerGhostLayers=True):
        if ghostLayers is True:
            ghostLayers = self.defaultGhostLayers
        elif ghostLayers is False:
            ghostLayers = 0
        elif isinstance(ghostLayers, str):
            ghostLayers = self.ghostLayersOfField(ghostLayers)

        if innerGhostLayers is True:
            innerGhostLayers = self.defaultGhostLayers
        elif innerGhostLayers is False:
            innerGhostLayers = 0
        elif isinstance(ghostLayers, str):
            ghostLayers = self.ghostLayersOfField(ghostLayers)

        prefix = self.GPU_DATA_PREFIX if gpu else ""
        if sliceObj is not None:
            yield from slicedBlockIteration(self.blocks, sliceObj, innerGhostLayers, ghostLayers,
                                            self.dim, prefix)
        else:
            yield from blockIteration(self.blocks, ghostLayers, self.dim, prefix)

    def gatherArray(self, name, sliceObj=None, allGather=False, ghostLayers=False):
        if ghostLayers is not False:
            warnings.warn("gatherArray with ghost layers is only supported in serial datahandling. "
                          "Array without ghost layers is returned")

        if sliceObj is None:
            sliceObj = tuple([slice(None, None, None)] * self.dim)
        if self.dim == 2:
            sliceObj = sliceObj[:2] + (0.5,) + sliceObj[2:]

        lastElement = sliceObj[3:]

        array = wlb.field.gatherField(self.blocks, name, sliceObj[:3], allGather)
        if array is None:
            return None

        if self.dim == 2:
            array = array[:, :, 0]
        if lastElement and self.fields[name].indexDimensions > 0:
            array = array[..., lastElement[0]]
        if self.fields[name].indexDimensions == 0:
            array = array[..., 0]

        return array

    def _normalizeArrShape(self, arr, indexDimensions):
        if indexDimensions == 0:
            arr = arr[..., 0]
        if self.dim == 2:
            arr = arr[:, :, 0]
        return arr

    def runKernel(self, kernelFunc, *args, **kwargs):
        if kernelFunc.ast.backend == 'gpucuda':
            nameMap = self._fieldNameToGpuDataName
            toArray = wlb.cuda.toGpuArray
        else:
            nameMap = self._fieldNameToCpuDataName
            toArray = wlb.field.toArray
        dataUsedInKernel = [(nameMap[p.fieldName], self.fields[p.fieldName])
                            for p in kernelFunc.parameters if p.isFieldPtrArgument and p.fieldName not in kwargs]
        for block in self.blocks:
            fieldArgs = {}
            for dataName, f in dataUsedInKernel:
                arr = toArray(block[dataName], withGhostLayers=[True, True, self.dim == 3])
                arr = self._normalizeArrShape(arr, f.indexDimensions)
                fieldArgs[f.name] = arr
            fieldArgs.update(kwargs)
            kernelFunc(*args, **fieldArgs)

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
            print("trying to transfer ", self.GPU_DATA_PREFIX + name)
            wlb.cuda.copyFieldToGpu(self.blocks, self.GPU_DATA_PREFIX + name, name)

    def isOnGpu(self, name):
        return (name, self.GPU_DATA_PREFIX + name) in self._cpuGpuPairs

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
        return self.synchronizationFunction(names, stencil, 'cpu',  buffered,)

    def synchronizationFunctionGPU(self, names, stencil=None, buffered=True, **kwargs):
        return self.synchronizationFunction(names, stencil, 'gpu', buffered)

    def synchronizationFunction(self, names, stencil=None, target='cpu', buffered=True):
        if target is None:
            target = self.defaultTarget

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

    def reduceFloatSequence(self, sequence, operation, allReduce=False):
        if allReduce:
            return np.array(wlb.mpi.allreduceReal(sequence, self._reduceMap[operation.lower()]))
        else:
            return np.array(wlb.mpi.reduceReal(sequence, self._reduceMap[operation.lower()]))

    def reduceIntSequence(self, sequence, operation, allReduce=False):
        if allReduce:
            return np.array(wlb.mpi.allreduceInt(sequence, self._reduceMap[operation.lower()]))
        else:
            return np.array(wlb.mpi.reduceInt(sequence, self._reduceMap[operation.lower()]))

    def vtkWriter(self, fileName, dataNames, ghostLayers=False):
        if ghostLayers is False:
            ghostLayers = 0
        if ghostLayers is True:
            ghostLayers = min(self.ghostLayersOfField(n) for n in dataNames)
        fileName = "%s_%02d" % (fileName, ParallelDataHandling.VTK_COUNTER)
        ParallelDataHandling.VTK_COUNTER += 1
        output = wlb.vtk.makeOutput(self.blocks, fileName, ghostLayers=ghostLayers)
        for n in dataNames:
            output.addCellDataWriter(wlb.field.createVTKWriter(self.blocks, n))
        return output

    def vtkWriterFlags(self, fileName, dataName, masksToName, ghostLayers=False):
        if ghostLayers is False:
            ghostLayers = 0
        if ghostLayers is True:
            ghostLayers = self.ghostLayersOfField(dataName)

        output = wlb.vtk.makeOutput(self.blocks, fileName, ghostLayers=ghostLayers)
        for mask, name in masksToName.items():
            w = wlb.field.createBinarizationVTKWriter(self.blocks, dataName, mask, name)
            output.addCellDataWriter(w)
        return output

