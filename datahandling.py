import numpy as np
from abc import ABC, abstractmethod, abstractproperty
from pystencils import Field, makeSlice
from pystencils.parallel.blockiteration import BlockIterationInfo
from pystencils.slicing import normalizeSlice, removeGhostLayers
from pystencils.utils import DotDict

try:
    import pycuda.gpuarray as gpuarray
except ImportError:
    gpuarray = None


class DataHandling(ABC):
    """
    Manages the storage of arrays and maps them to a symbolic field.
    Two versions are available: a simple, pure Python implementation for single node
    simulations :py:class:SerialDataHandling and a distributed version using waLBerla in :py:class:ParallelDataHandling

    Keep in mind that the data can be distributed, so use the 'access' method whenever possible and avoid the
    'gather' function that has collects (parts of the) distributed data on a single process.
    """

    # ---------------------------- Adding and accessing data -----------------------------------------------------------

    @property
    @abstractmethod
    def dim(self):
        """Dimension of the domain, either 2 or 3"""
        pass

    @abstractmethod
    def add(self, name, fSize=1, dtype=np.float64, latexName=None, ghostLayers=None, layout=None, cpu=True, gpu=False):
        """
        Adds a (possibly distributed) array to the handling that can be accessed using the given name.
        For each array a symbolic field is available via the 'fields' dictionary

        :param name: unique name that is used to access the field later
        :param fSize: shape of the dim+1 coordinate. DataHandling supports zero or one index dimensions, i.e. scalar
                      fields and vector fields. This parameter gives the shape of the index dimensions. The default
                      value of 1 means no index dimension
        :param dtype: data type of the array as numpy data type
        :param latexName: optional, name of the symbolic field, if not given 'name' is used
        :param ghostLayers: number of ghost layers - if not specified a default value specified in the constructor
                            is used
        :param layout: memory layout of array, either structure of arrays 'SoA' or array of structures 'AoS'.
                       this is only important if fSize > 1
        :param cpu: allocate field on the CPU
        :param gpu: allocate field on the GPU
        """
        pass

    @abstractmethod
    def addLike(self, name, nameOfTemplateField, latexName=None, cpu=True, gpu=False):
        """
        Adds an array with the same parameters (number of ghost layers, fSize, dtype) as existing array
        :param name: name of new array
        :param nameOfTemplateField: name of array that is used as template
        :param latexName: see 'add' method
        :param cpu: see 'add' method
        :param gpu: see 'add' method
        """
        pass

    @property
    @abstractmethod
    def fields(self):
        """Dictionary mapping data name to symbolic pystencils field - use this to create pystencils kernels"""
        pass

    @abstractmethod
    def access(self, name, sliceObj=None, innerGhostLayers=None, outerGhostLayers=0):
        """
        Generator yielding locally stored sub-arrays together with information about their place in the global domain

        :param name: name of data to access
        :param sliceObj: optional rectangular sub-region to access
        :param innerGhostLayers: how many inner (not at domain border) ghost layers to include
        :param outerGhostLayers: how many ghost layers at the domain border to include
        Yields a numpy array with local part of data and a BlockIterationInfo object containing geometric information
        """
        pass

    @abstractmethod
    def gather(self, name, sliceObj=None, allGather=False):
        """
        Gathers part of the domain on a local process. Whenever possible use 'access' instead, since this method copies
        the distributed data to a single process which is inefficient and may exhaust the available memory

        :param name: name of the array to gather
        :param sliceObj: slice expression of the rectangular sub-part that should be gathered
        :param allGather: if False only the root process receives the result, if True all processes
        :return: generator expression yielding the gathered field, the gathered field does not include any ghost layers
        """
        pass

    # ------------------------------- CPU/GPU transfer -----------------------------------------------------------------

    @abstractmethod
    def toCpu(self, name):
        """Copies GPU data of array with specified name to CPU.
        Works only if 'cpu=True' and 'gpu=True' has been used in 'add' method"""
        pass

    @abstractmethod
    def toGpu(self, name):
        """Copies GPU data of array with specified name to GPU.
        Works only if 'cpu=True' and 'gpu=True' has been used in 'add' method"""
        pass

    @abstractmethod
    def allToCpu(self, name):
        """Copies data from GPU to CPU for all arrays that have a CPU and a GPU representation"""
        pass

    @abstractmethod
    def allToGpu(self, name):
        """Copies data from CPU to GPU for all arrays that have a CPU and a GPU representation"""
        pass


class SerialDataHandling(DataHandling):

    class _PassThroughContextManager:
        def __init__(self, arr):
            self.arr = arr

        def __enter__(self, *args, **kwargs):
            return self.arr

    def __init__(self, domainSize, defaultGhostLayers=1, defaultLayout='SoA'):
        """
        Creates a data handling for single node simulations

        :param domainSize: size of the spatial domain as tuple
        :param defaultGhostLayers: nr of ghost layers used if not specified in add() method
        :param defaultLayout: layout used if no layout is given to add() method
        """
        self._domainSize = tuple(domainSize)
        self.defaultGhostLayers = defaultGhostLayers
        self.defaultLayout = defaultLayout
        self._fields = DotDict()
        self.cpuArrays = DotDict()
        self.gpuArrays = DotDict()
        self._fieldInformation = {}

    @property
    def dim(self):
        return len(self._domainSize)

    @property
    def fields(self):
        return self._fields

    def add(self, name, fSize=1, dtype=np.float64, latexName=None, ghostLayers=None, layout=None, cpu=True, gpu=False):
        if ghostLayers is None:
            ghostLayers = self.defaultGhostLayers
        if layout is None:
            layout = self.defaultLayout
        if latexName is None:
            latexName = name

        assert layout in ('SoA', 'AoS')

        kwargs = {
            'shape': tuple(s + 2 * ghostLayers for s in self._domainSize),
            'dtype': dtype,
            'order': 'c' if layout == 'AoS' else 'f',
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
        else:
            indexDimensions = 0
        if cpu:
            if name in self.cpuArrays:
                raise ValueError("CPU Field with this name already exists")
            self.cpuArrays[name] = np.empty(**kwargs)
        if gpu:
            if name in self.gpuArrays:
                raise ValueError("GPU Field with this name already exists")

            self.gpuArrays[name] = gpuarray.empty(**kwargs)

        assert all(f.name != latexName for f in self.fields.values()), "Symbolic field with this name already exists"
        self.fields[name] = Field.createFixedSize(latexName, shape=kwargs['shape'], indexDimensions=indexDimensions,
                                                  dtype=kwargs['dtype'], layout=kwargs['order'])

    def addLike(self, name, nameOfTemplateField, latexName=None, cpu=True, gpu=False):
        self.add(name,latexName=latexName, cpu=cpu, gpu=gpu, **self._fieldInformation[nameOfTemplateField])

    def access(self, name, sliceObj=None, outerGhostLayers=0, **kwargs):
        if sliceObj is None:
            sliceObj = [slice(None, None)] * self.dim
        arr = self.cpuArrays[name]
        glToRemove = self._fieldInformation[name]['ghostLayers'] - outerGhostLayers
        assert glToRemove >= 0
        arr = removeGhostLayers(arr, indexDimensions=self.fields[name].indexDimensions, ghostLayers=glToRemove)
        sliceObj = normalizeSlice(sliceObj, arr.shape[:self.dim])
        yield arr[sliceObj], BlockIterationInfo(None, tuple(s.start for s in sliceObj), sliceObj)

    def gather(self, name, sliceObj=None, **kwargs):
        gls = self._fieldInformation[name]['ghostLayers']
        arr = self.cpuArrays[name]
        arr = removeGhostLayers(arr, indexDimensions=self.fields[name].indexDimensions, ghostLayers=gls)

        if sliceObj is not None:
            arr = arr[sliceObj]
        yield arr

    def swap(self, name1, name2, gpu=False):
        if not gpu:
            self.cpuArrays[name1], self.cpuArrays[name2] = self.cpuArrays[name2], self.cpuArrays[name1]
        else:
            self.gpuArrays[name1], self.gpuArrays[name2] = self.gpuArrays[name2], self.gpuArrays[name1]

    def allToCpu(self):
        for name in self.cpuArrays.keys() & self.gpuArrays.keys():
            self.toCpu(name)

    def allToGpu(self):
        for name in self.cpuArrays.keys() & self.gpuArrays.keys():
            self.toGpu(name)

    def toCpu(self, name):
        self.gpuArrays[name].get(self.cpuArrays[name])

    def toGpu(self, name):
        self.gpuArrays[name].set(self.cpuArrays[name])
