import numpy as np
from pystencils import Field, makeSlice
from pystencils.datahandling import DataHandling, FlagArray, WalberlaFlagInterface
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
        if self._dim == 2:
            assert self.blocks.getDomainCellBB().size[2] == 1

    @property
    def dim(self):
        return self._dim

    @property
    def fields(self):
        return self._fields

    def add(self, name, fSize=1, dtype=np.float64, latexName=None, ghostLayers=None, layout=None, cpu=True, gpu=False):
        return self._add(name, fSize, dtype, latexName, ghostLayers, layout, cpu, gpu, flagField=False)

    def addLike(self, name, nameOfTemplateField, latexName=None, cpu=True, gpu=False):
        assert not self._fieldInformation[nameOfTemplateField]['flagField']
        self._add(name,latexName=latexName, cpu=cpu, gpu=gpu, **self._fieldInformation[nameOfTemplateField])

    def addFlagArray(self, name, dtype=np.int32, latexName=None, ghostLayers=None):
        return self._add(name, dtype=dtype, latexName=latexName, ghostLayers=ghostLayers, flagField=True)

    def swap(self, name1, name2, gpu=False):
        if gpu:
            name1 = self.GPU_DATA_PREFIX + name1
            name2 = self.GPU_DATA_PREFIX + name2
        for block in self.blocks:
            block[name1].swapDataPointers(block[name2])

    def access(self, name, sliceObj=None, innerGhostLayers=None, outerGhostLayers=0):
        fieldInfo = self._fieldInformation[name]
        with self.accessWrapper(name):
            if innerGhostLayers is None:
                innerGhostLayers = fieldInfo['ghostLayers']

            if outerGhostLayers is None:
                outerGhostLayers = fieldInfo['ghostLayers']

            for iterInfo in slicedBlockIteration(self.blocks, sliceObj, innerGhostLayers, outerGhostLayers):
                arr = wlb.field.toArray(iterInfo.block[name], withGhostLayers=innerGhostLayers)[iterInfo.localSlice]
                if fieldInfo['flagField']:
                    arr = FlagArray(arr, WalberlaFlagInterface(iterInfo.block[name]))
                if self.fields[name].indexDimensions == 0:
                    arr = arr[..., 0]
                if self.dim == 2:
                    arr = arr[:, :, 0]
                yield arr, iterInfo

    def gather(self, name, sliceObj=None, allGather=False):
        with self.accessWrapper(name):
            if sliceObj is None:
                sliceObj = makeSlice[:, :, :]
            for array in wlb.field.gatherGenerator(self.blocks, name, sliceObj, allGather):
                if self.fields[name].indexDimensions == 0:
                    array = array[..., 0]
                if self.dim == 2:
                    array = array[:, :, 0]
                yield array

    def toCpu(self, name):
        wlb.cuda.copyFieldToCpu(self.blocks, self.GPU_DATA_PREFIX + name, name)

    def toGpu(self, name):
        wlb.cuda.copyFieldToGpu(self.blocks, self.GPU_DATA_PREFIX + name, name)

    def allToCpu(self):
        for cpuName, gpuName in self._cpuGpuPairs:
            wlb.cuda.copyFieldToCpu(self.blocks, gpuName, cpuName)

    def allToGpu(self):
        for cpuName, gpuName in self._cpuGpuPairs:
            wlb.cuda.copyFieldToGpu(self.blocks, gpuName, cpuName)

    def _add(self, name, fSize=1, dtype=np.float64, latexName=None, ghostLayers=None, layout=None,
            cpu=True, gpu=False, flagField=False):
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
                                        'dtype': dtype,
                                        'flagField': flagField}

        layoutMap = {'fzyx': wlb.field.Layout.fzyx, 'zyxf': wlb.field.Layout.zyxf,
                     'SoA': wlb.field.Layout.fzyx,  'AoS': wlb.field.Layout.zyxf}

        if flagField:
            assert not gpu
            assert np.dtype(dtype).kind in ('u', 'i'), "FlagArrays can only be created from integer arrays"
            nrOfBits = np.dtype(dtype).itemsize * 8
            wlb.field.addFlagFieldToStorage(self.blocks, name, nrOfBits, ghostLayers)
        else:
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
