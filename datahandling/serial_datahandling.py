import numpy as np
from lbmpy.stencils import getStencil
from pystencils import Field
from pystencils.field import layoutStringToTuple, spatialLayoutStringToTuple, createNumpyArrayWithLayout
from pystencils.parallel.blockiteration import Block, SerialBlock
from pystencils.slicing import normalizeSlice, removeGhostLayers
from pystencils.utils import DotDict
from pystencils.datahandling.datahandling_interface import DataHandling

try:
    import pycuda.gpuarray as gpuarray
except ImportError:
    gpuarray = None


class SerialDataHandling(DataHandling):

    class _PassThroughContextManager:
        def __init__(self, arr):
            self.arr = arr

        def __enter__(self, *args, **kwargs):
            return self.arr

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
                                                  dtype=kwargs['dtype'], layout=layout)
        self._fieldLatexNameToDataName[latexName] = name

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

    def iterate(self, sliceObj=None, gpu=False, ghostLayers=True):
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

    def gatherArray(self, name, sliceObj=None, **kwargs):
        gls = self._fieldInformation[name]['ghostLayers']
        arr = self.cpuArrays[name]
        indDimensions = self.fields[name].indexDimensions
        arr = removeGhostLayers(arr, indexDimensions=indDimensions, ghostLayers=gls)

        if sliceObj is not None:
            sliceObj = normalizeSlice(sliceObj, arr.shape[:-indDimensions])
            arr = arr[sliceObj]
        yield arr

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

    def synchronizationFunctionCPU(self, names, stencilName=None, **kwargs):
        return self._synchronizationFunctor(names, stencilName, 'cpu')

    def synchronizationFunctionGPU(self, names, stencilName=None, **kwargs):
        return self._synchronizationFunctor(names, stencilName, 'gpu')

    def _synchronizationFunctor(self, names, stencil, target):
        if stencil is None:
            stencil = 'D3Q27' if self.dim == 3 else 'D2Q9'

        assert stencil in ("D2Q9", 'D3Q27'), "Serial scenario support only D2Q9 or D3Q27 for periodicity sync"

        assert target in ('cpu', 'gpu')
        if not hasattr(names, '__len__') or type(names) is str:
            names = [names]

        filteredStencil = []
        for direction in getStencil(stencil):
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
