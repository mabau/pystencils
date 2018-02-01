import numpy as np
from abc import ABC, abstractmethod


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

    @property
    @abstractmethod
    def shape(self):
        """Shape of outer bounding box"""

    @property
    @abstractmethod
    def periodicity(self):
        """Returns tuple of booleans for x,y,(z) directions with True if domain is periodic in that direction"""

    @abstractmethod
    def addArray(self, name, fSize=1, dtype=np.float64, latexName=None, ghostLayers=None, layout=None, cpu=True, gpu=False):
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

    @abstractmethod
    def hasData(self, name):
        """
        Returns true if a field or custom data element with this name was added
        """

    @abstractmethod
    def addArrayLike(self, name, nameOfTemplateField, latexName=None, cpu=True, gpu=False):
        """
        Adds an array with the same parameters (number of ghost layers, fSize, dtype) as existing array
        :param name: name of new array
        :param nameOfTemplateField: name of array that is used as template
        :param latexName: see 'add' method
        :param cpu: see 'add' method
        :param gpu: see 'add' method
        """

    @abstractmethod
    def addCustomData(self, name, cpuCreationFunction,
                      gpuCreationFunction=None, cpuToGpuTransferFunc=None, gpuToCpuTransferFunc=None):
        """
        Adds custom (non-array) data to domain
        :param name: name to access data
        :param cpuCreationFunction: function returning a new instance of the data that should be stored
        :param gpuCreationFunction: optional, function returning a new instance, stored on GPU
        :param cpuToGpuTransferFunc: function that transfers cpu to gpu version, getting two parameters (gpuInstance, cpuInstance)
        :param gpuToCpuTransferFunc: function that transfers gpu to cpu version, getting two parameters (gpuInstance, cpuInstance)
        :return:
        """

    def addCustomClass(self, name, classObj, cpu=True, gpu=False):
        self.addCustomData(name,
                           cpuCreationFunction=classObj if cpu else None,
                           gpuCreationFunction=classObj if gpu else None,
                           cpuToGpuTransferFunc=classObj.toGpu if cpu and gpu and hasattr(classObj, 'toGpu') else None,
                           gpuToCpuTransferFunc=classObj.toCpu if cpu and gpu and hasattr(classObj, 'toCpu') else None)

    @property
    @abstractmethod
    def fields(self):
        """Dictionary mapping data name to symbolic pystencils field - use this to create pystencils kernels"""

    @abstractmethod
    def ghostLayersOfField(self, name):
        """Returns the number of ghost layers for a specific field/array"""

    @abstractmethod
    def iterate(self, sliceObj=None, gpu=False, ghostLayers=None, innerGhostLayers=True):
        """
        Iterate over local part of potentially distributed data structure.
        """

    @abstractmethod
    def gatherArray(self, name, sliceObj=None, allGather=False, ghostLayers=False):
        """
        Gathers part of the domain on a local process. Whenever possible use 'access' instead, since this method copies
        the distributed data to a single process which is inefficient and may exhaust the available memory

        :param name: name of the array to gather
        :param sliceObj: slice expression of the rectangular sub-part that should be gathered
        :param allGather: if False only the root process receives the result, if True all processes
        :param ghostLayers: number of outer ghost layers to include (only available for serial data handlings)
        :return: gathered field that does not include any ghost layers, or None if gathered on another process
        """

    @abstractmethod
    def runKernel(self, kernelFunc, *args, **kwargs):
        """
        Runs a compiled pystencils kernel using the arrays stored in the DataHandling class for all array parameters
        Additional passed arguments are directly passed to the kernel function and override possible parameters from
        the DataHandling
        """

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


    @abstractmethod
    def vtkWriter(self, fileName, dataNames, ghostLayers=False):
        """VTK output for one or multiple arrays
        :param fileName: base file name without extension for the VTK output
        :param dataNames: list of array names that should be included in the vtk output
        :param ghostLayers: true if ghost layer information should be written out as well
        :return: a function that can be called with an integer time step to write the current state
                i.e vtkWriter('someFile', ['velocity', 'density']) (1)
        """
    @abstractmethod
    def vtkWriterFlags(self, fileName, dataName, masksToName, ghostLayers=False):
        """VTK output for an unsigned integer field, where bits are intepreted as flags
        :param fileName: see vtkWriter
        :param dataName: name of an array with uint type
        :param masksToName: dictionary mapping integer masks to a name in the output
        :param ghostLayers: see vtkWriter
        :returns: functor that can be called with time step
         """

    # ------------------------------- Communication --------------------------------------------------------------------

    def synchronizationFunctionCPU(self, names, stencil=None, **kwargs):
        """
        Synchronizes ghost layers for distributed arrays - for serial scenario this has to be called
        for correct periodicity handling
        :param names: what data to synchronize: name of array or sequence of names
        :param stencil: stencil as string defining which neighbors are synchronized e.g. 'D2Q9', 'D3Q19'
                        if None, a full synchronization (i.e. D2Q9 or D3Q27) is done
        :param kwargs: implementation specific, optional optimization parameters for communication
        :return: function object to run the communication
        """

    def synchronizationFunctionGPU(self, names, stencil=None, **kwargs):
        """
        Synchronization of GPU fields, for documentation see CPU version above
        """

    def reduceFloatSequence(self, sequence, operation, allReduce=False):
        """Takes a sequence of floating point values on each process and reduces it element wise to all
        processes (allReduce=True) or only to the root process (allReduce=False).
        Possible operations are 'sum', 'min', 'max'
        """

    def reduceIntSequence(self, sequence, operation, allReduce=False):
        """See function reduceFloatSequence - this is the same for integers"""
