import numpy as np
import itertools
from pystencils import Field
from pystencils.field import layoutStringToTuple, spatialLayoutStringToTuple, createNumpyArrayWithLayout
from pystencils.parallel.blockiteration import SerialBlock
from pystencils.slicing import normalizeSlice, removeGhostLayers
from pystencils.utils import DotDict
from pystencils.datahandling.datahandling_interface import DataHandling

try:
    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit
except ImportError:
    gpuarray = None


class SerialDataHandling(DataHandling):

    def __init__(self, domainSize, defaultGhostLayers=1, defaultLayout='SoA', periodicity=False):
        """
        Creates a data handling for single node simulations

        :param domainSize: size of the spatial domain as tuple
        :param defaultGhostLayers: nr of ghost layers used if not specified in add() method
        :param defaultLayout: layout used if no layout is given to add() method
        """
        super(SerialDataHandling, self).__init__()
        self._domainSize = tuple(domainSize)
        self.defaultGhostLayers = defaultGhostLayers
        self.defaultLayout = defaultLayout
        self._fields = DotDict()
        self._fieldLatexNameToDataName = {}
        self.cpuArrays = DotDict()
        self.gpuArrays = DotDict()
        self.customDataCpu = DotDict()
        self.customDataGpu = DotDict()
        self._customDataTransferFunctions = {}

        if periodicity is None or periodicity is False:
            periodicity = [False] * self.dim
        if periodicity is True:
            periodicity = [True] * self.dim

        self._periodicity = periodicity
        self._fieldInformation = {}

    @property
    def dim(self):
        return len(self._domainSize)

    @property
    def shape(self):
        return self._domainSize

    @property
    def periodicity(self):
        return self._periodicity

    @property
    def fields(self):
        return self._fields

    def ghostLayersOfField(self, name):
        return self._fieldInformation[name]['ghostLayers']

    def addArray(self, name, fSize=1, dtype=np.float64, latexName=None, ghostLayers=None, layout=None,
                 cpu=True, gpu=False):
        if ghostLayers is None:
            ghostLayers = self.defaultGhostLayers
        if layout is None:
            layout = self.defaultLayout
        if latexName is None:
            latexName = name

        kwargs = {
            'shape': tuple(s + 2 * ghostLayers for s in self._domainSize),
            'dtype': dtype,
        }
        self._fieldInformation[name] = {
            'ghostLayers': ghostLayers,
            'fSize': fSize,
            'layout': layout,
            'dtype': dtype,
        }

        if fSize > 1:
            kwargs['shape'] = kwargs['shape'] + (fSize,)
            indexDimensions = 1
            layoutTuple = layoutStringToTuple(layout, self.dim+1)
        else:
            indexDimensions = 0
            layoutTuple = spatialLayoutStringToTuple(layout, self.dim)

        # cpuArr is always created - since there is no createPycudaArrayWithLayout()
        cpuArr = createNumpyArrayWithLayout(layout=layoutTuple, **kwargs)
        if cpu:
            if name in self.cpuArrays:
                raise ValueError("CPU Field with this name already exists")
            self.cpuArrays[name] = cpuArr
        if gpu:
            if name in self.gpuArrays:
                raise ValueError("GPU Field with this name already exists")
            self.gpuArrays[name] = gpuarray.to_gpu(cpuArr)

        assert all(f.name != latexName for f in self.fields.values()), "Symbolic field with this name already exists"
        self.fields[name] = Field.createFixedSize(latexName, shape=kwargs['shape'], indexDimensions=indexDimensions,
                                                  dtype=kwargs['dtype'], layout=layoutTuple)
        self._fieldLatexNameToDataName[latexName] = name
        return self.fields[name]

    def addCustomData(self, name, cpuCreationFunction,
                      gpuCreationFunction=None, cpuToGpuTransferFunc=None, gpuToCpuTransferFunc=None):

        if cpuCreationFunction and gpuCreationFunction:
            if cpuToGpuTransferFunc is None or gpuToCpuTransferFunc is None:
                raise ValueError("For GPU data, both transfer functions have to be specified")
            self._customDataTransferFunctions[name] = (cpuToGpuTransferFunc, gpuToCpuTransferFunc)

        assert name not in self.customDataCpu
        if cpuCreationFunction:
            assert name not in self.cpuArrays
            self.customDataCpu[name] = cpuCreationFunction()

        if gpuCreationFunction:
            assert name not in self.gpuArrays
            self.customDataGpu[name] = gpuCreationFunction()

    def hasData(self, name):
        return name in self.fields

    def addArrayLike(self, name, nameOfTemplateField, latexName=None, cpu=True, gpu=False):
        self.addArray(name, latexName=latexName, cpu=cpu, gpu=gpu, **self._fieldInformation[nameOfTemplateField])

    def iterate(self, sliceObj=None, gpu=False, ghostLayers=True, innerGhostLayers=True):
        if ghostLayers is True:
            ghostLayers = self.defaultGhostLayers
        elif ghostLayers is False:
            ghostLayers = 0

        if sliceObj is None:
            sliceObj = (slice(None, None, None),) * self.dim
        sliceObj = normalizeSlice(sliceObj, tuple(s + 2 * ghostLayers for s in self._domainSize))
        sliceObj = tuple(s if type(s) is slice else slice(s, s+1, None) for s in sliceObj)

        arrays = self.gpuArrays if gpu else self.cpuArrays
        customDataDict = self.customDataGpu if gpu else self.customDataCpu
        iterDict = customDataDict.copy()
        for name, arr in arrays.items():
            fieldGls = self._fieldInformation[name]['ghostLayers']
            if fieldGls < ghostLayers:
                continue
            arr = removeGhostLayers(arr, indexDimensions=len(arr.shape) - self.dim, ghostLayers=fieldGls-ghostLayers)
            iterDict[name] = arr

        offset = tuple(s.start - ghostLayers for s in sliceObj)
        yield SerialBlock(iterDict, offset, sliceObj)

    def gatherArray(self, name, sliceObj=None, ghostLayers=False, **kwargs):
        glToRemove = self._fieldInformation[name]['ghostLayers']
        if isinstance(ghostLayers, int):
            glToRemove -= ghostLayers
        if ghostLayers is True:
            glToRemove = 0
        arr = self.cpuArrays[name]
        indDimensions = self.fields[name].indexDimensions
        arr = removeGhostLayers(arr, indexDimensions=indDimensions, ghostLayers=glToRemove)

        if sliceObj is not None:
            sliceObj = normalizeSlice(sliceObj, arr.shape[:-indDimensions] if indDimensions > 0 else arr.shape)
            sliceObj = tuple(s if type(s) is slice else slice(s, s + 1, None) for s in sliceObj)
            arr = arr[sliceObj]
        else:
            arr = arr.view()
        arr.flags.writeable = False
        return arr

    def swap(self, name1, name2, gpu=False):
        if not gpu:
            self.cpuArrays[name1], self.cpuArrays[name2] = self.cpuArrays[name2], self.cpuArrays[name1]
        else:
            self.gpuArrays[name1], self.gpuArrays[name2] = self.gpuArrays[name2], self.gpuArrays[name1]

    def allToCpu(self):
        for name in (self.cpuArrays.keys() & self.gpuArrays.keys()) | self._customDataTransferFunctions.keys():
            self.toCpu(name)

    def allToGpu(self):
        for name in (self.cpuArrays.keys() & self.gpuArrays.keys()) | self._customDataTransferFunctions.keys():
            self.toGpu(name)

    def runKernel(self, kernelFunc, *args, **kwargs):
        dataUsedInKernel = [self._fieldLatexNameToDataName[p.fieldName]
                            for p in kernelFunc.parameters if p.isFieldPtrArgument]
        arrays = self.gpuArrays if kernelFunc.ast.backend == 'gpucuda' else self.cpuArrays
        arrayParams = {name: arrays[name] for name in dataUsedInKernel}
        arrayParams.update(kwargs)
        kernelFunc(*args, **arrayParams)

    def toCpu(self, name):
        if name in self._customDataTransferFunctions:
            transferFunc = self._customDataTransferFunctions[name][1]
            transferFunc(self.customDataGpu[name], self.customDataCpu[name])
        else:
            self.gpuArrays[name].get(self.cpuArrays[name])

    def toGpu(self, name):
        if name in self._customDataTransferFunctions:
            transferFunc = self._customDataTransferFunctions[name][0]
            transferFunc(self.customDataGpu[name], self.customDataCpu[name])
        else:
            self.gpuArrays[name].set(self.cpuArrays[name])

    def isOnGpu(self, name):
        return name in self.gpuArrays

    def synchronizationFunctionCPU(self, names, stencilName=None, **kwargs):
        return self._synchronizationFunctor(names, stencilName, 'cpu')

    def synchronizationFunctionGPU(self, names, stencilName=None, **kwargs):
        return self._synchronizationFunctor(names, stencilName, 'gpu')

    def _synchronizationFunctor(self, names, stencil, target):
        assert target in ('cpu', 'gpu')
        if not hasattr(names, '__len__') or type(names) is str:
            names = [names]

        filteredStencil = []
        neighbors = [-1, 0, 1]

        if (stencil is None and self.dim == 2) or (stencil is not None and stencil.startswith('D2')):
            directions = itertools.product(*[neighbors] * 2)
        elif (stencil is None and self.dim == 3) or (stencil is not None and stencil.startswith('D3')):
            directions = itertools.product(*[neighbors] * 3)
        else:
            raise ValueError("Invalid stencil")

        for direction in directions:
            useDirection = True
            if direction == (0, 0) or direction == (0, 0, 0):
                useDirection = False
            for component, periodicity in zip(direction, self._periodicity):
                if not periodicity and component != 0:
                    useDirection = False
            if useDirection:
                filteredStencil.append(direction)

        resultFunctors = []
        for name in names:
            gls = self._fieldInformation[name]['ghostLayers']
            if len(filteredStencil) > 0:
                if target == 'cpu':
                    from pystencils.slicing import getPeriodicBoundaryFunctor
                    resultFunctors.append(getPeriodicBoundaryFunctor(filteredStencil, ghostLayers=gls))
                else:
                    from pystencils.gpucuda.periodicity import getPeriodicBoundaryFunctor
                    resultFunctors.append(getPeriodicBoundaryFunctor(filteredStencil, self._domainSize,
                                                                     indexDimensions=self.fields[name].indexDimensions,
                                                                     indexDimShape=self._fieldInformation[name]['fSize'],
                                                                     dtype=self.fields[name].dtype.numpyDtype,
                                                                     ghostLayers=gls))

        if target == 'cpu':
            def resultFunctor():
                for func in resultFunctors:
                    func(pdfs=self.cpuArrays[name])
        else:
            def resultFunctor():
                for func in resultFunctors:
                    func(pdfs=self.gpuArrays[name])

        return resultFunctor

    @staticmethod
    def reduceFloatSequence(sequence, operation, allReduce=False):
        return np.array(sequence)

    @staticmethod
    def reduceIntSequence(sequence):
        return np.array(sequence)

    def vtkWriter(self, fileName, dataNames, ghostLayers=False):
        from pystencils.vtk import imageToVTK

        def writer(step):
            fullFileName = "%s_%08d" % (fileName, step,)
            cellData = {}
            for name in dataNames:
                field = self._getFieldWithGivenNumberOfGhostLayers(name, ghostLayers)
                if self.dim == 2:
                    field = field[:, :, np.newaxis]
                if len(field.shape) == 3:
                    field = np.ascontiguousarray(field)
                elif len(field.shape) == 4:
                    field = [np.ascontiguousarray(field[..., i]) for i in range(field.shape[-1])]
                    if len(field) == 2:
                        field.append(np.zeros_like(field[0]))
                    field = tuple(field)
                else:
                    assert False
                cellData[name] = field
            imageToVTK(fullFileName, cellData=cellData)
        return writer

    def vtkWriterFlags(self, fileName, dataName, masksToName, ghostLayers=False):
        from pystencils.vtk import imageToVTK

        def writer(step):
            fullFileName = "%s_%08d" % (fileName, step,)
            field = self._getFieldWithGivenNumberOfGhostLayers(dataName, ghostLayers)
            if self.dim == 2:
                field = field[:, :, np.newaxis]
            cellData = {name: np.ascontiguousarray(np.bitwise_and(field, mask) > 0, dtype=np.uint8)
                        for mask, name in masksToName.items()}
            imageToVTK(fullFileName, cellData=cellData)

        return writer

    def _getFieldWithGivenNumberOfGhostLayers(self, name, ghostLayers):
        actualGhostLayers = self.ghostLayersOfField(name)
        if ghostLayers is True:
            ghostLayers = actualGhostLayers

        glToRemove = actualGhostLayers - ghostLayers
        indDims = 1 if self._fieldInformation[name]['fSize'] > 1 else 0
        return removeGhostLayers(self.cpuArrays[name], indDims, glToRemove)

