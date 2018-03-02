import numpy as np
import sympy as sp
from pystencils import Field, TypedSymbol, createIndexedKernel
from pystencils.backends.cbackend import CustomCppCode
from pystencils.boundaries.createindexlist import numpyDataTypeForBoundaryObject, createBoundaryIndexArray
from pystencils.cache import memorycache
from pystencils.data_types import createType


class FlagInterface:
    FLAG_DTYPE = np.uint32

    def __init__(self, dataHandling, flagFieldName):
        self.flagFieldName = flagFieldName
        self.domainFlag = self.FLAG_DTYPE(1 << 0)
        self._nextFreeFlag = 1
        self.dataHandling = dataHandling

        # Add flag field to data handling if it does not yet exist
        if dataHandling.hasData(self.flagFieldName):
            raise ValueError("There is already a boundary handling registered at the data handling."
                             "If you want to add multiple handlings, choose a different name.")

        dataHandling.addArray(self.flagFieldName, dtype=self.FLAG_DTYPE, cpu=True, gpu=False)
        ffGhostLayers = dataHandling.ghostLayersOfField(self.flagFieldName)
        for b in dataHandling.iterate(ghostLayers=ffGhostLayers):
            b[self.flagFieldName].fill(self.domainFlag)

    def allocateNextFlag(self):
        result = self.FLAG_DTYPE(1 << self._nextFreeFlag)
        self._nextFreeFlag += 1
        return result


class BoundaryHandling:

    def __init__(self, dataHandling, fieldName, stencil, name="boundaryHandling", flagInterface=None,
                 target='cpu', openMP=True):
        assert dataHandling.hasData(fieldName)

        self._dataHandling = dataHandling
        self._fieldName = fieldName
        self._indexArrayName = name + "IndexArrays"
        self._target = target
        self._openMP = openMP
        self._boundaryObjectToBoundaryInfo = {}
        self.stencil = stencil
        self._dirty = True
        self.flagInterface = flagInterface if flagInterface is not None else FlagInterface(dataHandling, name + "Flags")

        gpu = self._target == 'gpu'
        dataHandling.addCustomClass(self._indexArrayName, self.IndexFieldBlockData, cpu=True, gpu=gpu)

    @property
    def dataHandling(self):
        return self._dataHandling

    def getFlag(self, boundaryObj):
        return self._boundaryObjectToBoundaryInfo[boundaryObj].flag

    @property
    def shape(self):
        return self._dataHandling.shape

    @property
    def dim(self):
        return self._dataHandling.dim

    @property
    def boundaryObjects(self):
        return tuple(self._boundaryObjectToName.keys())

    @property
    def flagArrayName(self):
        return self.flagInterface.flagFieldName

    def getBoundaryNameToFlagDict(self):
        result = {bObj.name: bInfo.flag for bObj, bInfo in self._boundaryObjectToBoundaryInfo.items()}
        result['domain'] = self.flagInterface.domainFlag
        return result

    def getMask(self, sliceObj, boundaryObj, inverse=False):
        if isinstance(boundaryObj, str) and boundaryObj.lower() == 'domain':
            flag = self.flagInterface.domainFlag
        else:
            flag = self._boundaryObjectToBoundaryInfo[boundaryObj].flag

        arr = self.dataHandling.gatherArray(self.flagArrayName, sliceObj)
        if arr is None:
            return None
        else:
            result = np.bitwise_and(arr, flag)
            if inverse:
                result = np.logical_not(result)
            return result

    def setBoundary(self, boundaryObject, sliceObj=None, maskCallback=None, ghostLayers=True, innerGhostLayers=True,
                    replace=True):
        """
        Sets boundary using either a rectangular slice, a boolean mask or a combination of both

        :param boundaryObject: instance of a boundary object that should be set
        :param sliceObj: a slice object (can be created with makeSlice[]) that selects a part of the domain where
                          the boundary should be set. If none, the complete domain is selected which makes only sense
                          if a maskCallback is passed. The slice can have ':' placeholders, which are interpreted
                          depending on the 'includeGhostLayers' parameter i.e. if it is True, the slice extends
                          into the ghost layers
        :param maskCallback: callback function getting x,y (z) parameters of the cell midpoints and returning a
                             boolean mask with True entries where boundary cells should be set.
                             The x, y, z arrays have 2D/3D shape such that they can be used directly
                             to create the boolean return array. i.e return x < 10 sets boundaries in cells with
                             midpoint x coordinate smaller than 10.
        :param ghostLayers see DataHandling.iterate()
        """
        if isinstance(boundaryObject, str) and boundaryObject.lower() == 'domain':
            flag = self.flagInterface.domainFlag
        else:
            flag = self._addBoundary(boundaryObject)

        for b in self._dataHandling.iterate(sliceObj, ghostLayers=ghostLayers, innerGhostLayers=innerGhostLayers):
            flagArr = b[self.flagInterface.flagFieldName]
            if maskCallback is not None:
                mask = maskCallback(*b.midpointArrays)
                if replace:
                    flagArr[mask] = flag
                else:
                    np.bitwise_or(flagArr, flag, where=mask, out=flagArr)
                    np.bitwise_and(flagArr, ~self.flagInterface.domainFlag, where=mask, out=flagArr)
            else:
                if replace:
                    flagArr.fill(flag)
                else:
                    np.bitwise_or(flagArr, flag, out=flagArr)
                    np.bitwise_and(flagArr, ~self.flagInterface.domainFlag, out=flagArr)

        self._dirty = True

        return flag

    def setBoundaryWhereFlagIsSet(self, boundaryObject, flag):
        self._addBoundary(boundaryObject, flag)
        self._dirty = True
        return flag

    def prepare(self):
        if not self._dirty:
            return
        self._createIndexFields()
        self._dirty = False

    def triggerReinitializationOfBoundaryData(self, **kwargs):
        if self._dirty:
            self.prepare()
        else:
            ffGhostLayers = self._dataHandling.ghostLayersOfField(self.flagInterface.flagFieldName)
            for b in self._dataHandling.iterate(ghostLayers=ffGhostLayers):
                for bObj, setter in b[self._indexArrayName].boundaryObjectToDataSetter.items():
                    self._boundaryDataInitialization(bObj, setter, **kwargs)

    def __call__(self, **kwargs):
        if self._dirty:
            self.prepare()

        for b in self._dataHandling.iterate(gpu=self._target == 'gpu'):
            for bObj, idxArr in b[self._indexArrayName].boundaryObjectToIndexList.items():
                kwargs[self._fieldName] = b[self._fieldName]
                kwargs['indexField'] = idxArr
                dataUsedInKernel = (p.fieldName
                                    for p in self._boundaryObjectToBoundaryInfo[bObj].kernel.parameters
                                    if p.isFieldPtrArgument and p.fieldName not in kwargs)
                kwargs.update({name: b[name] for name in dataUsedInKernel})

                self._boundaryObjectToBoundaryInfo[bObj].kernel(**kwargs)

    def geometryToVTK(self, fileName='geometry', boundaries='all', ghostLayers=False):
        """
        Writes a VTK field where each cell with the given boundary is marked with 1, other cells are 0
        This can be used to display the simulation geometry in Paraview
        :param fileName: vtk filename
        :param boundaries: boundary object, or special string 'domain' for domain cells or special string 'all' for all
                         boundary conditions.
                         can also  be a sequence, to write multiple boundaries to VTK file
        :param ghostLayers: number of ghost layers to write, or True for all, False for none
        """
        if boundaries == 'all':
            boundaries = list(self._boundaryObjectToBoundaryInfo.keys()) + ['domain']
        elif not hasattr(boundaries, "__len__"):
            boundaries = [boundaries]

        masksToName = {}
        for b in boundaries:
            if b == 'domain':
                masksToName[self.flagInterface.domainFlag] = 'domain'
            else:
                masksToName[self._boundaryObjectToBoundaryInfo[b].flag] = b.name

        writer = self.dataHandling.vtkWriterFlags(fileName, self.flagInterface.flagFieldName,
                                                  masksToName, ghostLayers=ghostLayers)
        writer(1)

    # ------------------------------ Implementation Details ------------------------------------------------------------

    def _addBoundary(self, boundaryObject, flag=None):
        if boundaryObject not in self._boundaryObjectToBoundaryInfo:
            symbolicIndexField = Field.createGeneric('indexField', spatialDimensions=1,
                                                     dtype=numpyDataTypeForBoundaryObject(boundaryObject, self.dim))
            ast = self._createBoundaryKernel(self._dataHandling.fields[self._fieldName],
                                             symbolicIndexField, boundaryObject)
            if flag is None:
                flag = self.flagInterface.allocateNextFlag()
            boundaryInfo = self.BoundaryInfo(boundaryObject, flag=flag, kernel=ast.compile())
            self._boundaryObjectToBoundaryInfo[boundaryObject] = boundaryInfo
        return self._boundaryObjectToBoundaryInfo[boundaryObject].flag

    def _createBoundaryKernel(self, symbolicField, symbolicIndexField, boundaryObject):
        return createBoundaryKernel(symbolicField, symbolicIndexField, self.stencil, boundaryObject,
                                    target=self._target, openMP=self._openMP)

    def _createIndexFields(self):
        dh = self._dataHandling
        ffGhostLayers = dh.ghostLayersOfField(self.flagInterface.flagFieldName)
        for b in dh.iterate(ghostLayers=ffGhostLayers):
            flagArr = b[self.flagInterface.flagFieldName]
            pdfArr = b[self._fieldName]
            indexArrayBD = b[self._indexArrayName]
            indexArrayBD.clear()
            for bInfo in self._boundaryObjectToBoundaryInfo.values():
                idxArr = createBoundaryIndexArray(flagArr, self.stencil, bInfo.flag, self.flagInterface.domainFlag,
                                                  bInfo.boundaryObject, ffGhostLayers)
                if idxArr.size == 0:
                    continue

                boundaryDataSetter = BoundaryDataSetter(idxArr, b.offset, self.stencil, ffGhostLayers, pdfArr)
                indexArrayBD.boundaryObjectToIndexList[bInfo.boundaryObject] = idxArr
                indexArrayBD.boundaryObjectToDataSetter[bInfo.boundaryObject] = boundaryDataSetter
                self._boundaryDataInitialization(bInfo.boundaryObject, boundaryDataSetter)

    def _boundaryDataInitialization(self, boundaryObject, boundaryDataSetter, **kwargs):
        if boundaryObject.additionalDataInitCallback:
            boundaryObject.additionalDataInitCallback(boundaryDataSetter, **kwargs)
        if self._target == 'gpu':
            self._dataHandling.toGpu(self._indexArrayName)

    class BoundaryInfo(object):
        def __init__(self, boundaryObject, flag, kernel):
            self.boundaryObject = boundaryObject
            self.flag = flag
            self.kernel = kernel

    class IndexFieldBlockData:
        def __init__(self, *args, **kwargs):
            self.boundaryObjectToIndexList = {}
            self.boundaryObjectToDataSetter = {}

        def clear(self):
            self.boundaryObjectToIndexList.clear()
            self.boundaryObjectToDataSetter.clear()

        @staticmethod
        def toCpu(gpuVersion, cpuVersion):
            gpuVersion = gpuVersion.boundaryObjectToIndexList
            cpuVersion = cpuVersion.boundaryObjectToIndexList
            for obj, cpuArr in cpuVersion.values():
                gpuVersion[obj].get(cpuArr)

        @staticmethod
        def toGpu(gpuVersion, cpuVersion):
            from pycuda import gpuarray
            gpuVersion = gpuVersion.boundaryObjectToIndexList
            cpuVersion = cpuVersion.boundaryObjectToIndexList
            for obj, cpuArr in cpuVersion.items():
                if obj not in gpuVersion:
                    gpuVersion[obj] = gpuarray.to_gpu(cpuArr)
                else:
                    gpuVersion[obj].set(cpuArr)


class BoundaryDataSetter:

    def __init__(self, indexArray, offset, stencil, ghostLayers, pdfArray):
        self.indexArray = indexArray
        self.offset = offset
        self.stencil = np.array(stencil)
        self.pdfArray = pdfArray.view()
        self.pdfArray.flags.writeable = False

        arrFieldNames = indexArray.dtype.names
        self.dim = 3 if 'z' in arrFieldNames else 2
        assert 'x' in arrFieldNames and 'y' in arrFieldNames and 'dir' in arrFieldNames, str(arrFieldNames)
        self.boundaryDataNames = set(self.indexArray.dtype.names) - set(['x', 'y', 'z', 'dir'])
        self.coordMap = {0: 'x', 1: 'y', 2: 'z'}
        self.ghostLayers = ghostLayers

    def nonBoundaryCellPositions(self, coord):
        assert coord < self.dim
        return self.indexArray[self.coordMap[coord]] + self.offset[coord] - self.ghostLayers + 0.5

    @memorycache()
    def linkOffsets(self):
        return self.stencil[self.indexArray['dir']]

    @memorycache()
    def linkPositions(self, coord):
        return self.nonBoundaryCellPositions(coord) + 0.5 * self.linkOffsets()[:, coord]

    @memorycache()
    def boundaryCellPositions(self, coord):
        return self.nonBoundaryCellPositions(coord) + self.linkOffsets()[:, coord]

    def __setitem__(self, key, value):
        if key not in self.boundaryDataNames:
            raise KeyError("Invalid boundary data name %s. Allowed are %s" % (key, self.boundaryDataNames))
        self.indexArray[key] = value

    def __getitem__(self, item):
        if item not in self.boundaryDataNames:
            raise KeyError("Invalid boundary data name %s. Allowed are %s" % (item, self.boundaryDataNames))
        return self.indexArray[item]


class BoundaryOffsetInfo(CustomCppCode):

    # --------------------------- Functions to be used by boundaries --------------------------

    @staticmethod
    def offsetFromDir(dirIdx, dim):
        return tuple([sp.IndexedBase(symbol, shape=(1,))[dirIdx]
                      for symbol in BoundaryOffsetInfo._offsetSymbols(dim)])

    @staticmethod
    def invDir(dirIdx):
        return sp.IndexedBase(BoundaryOffsetInfo.INV_DIR_SYMBOL, shape=(1,))[dirIdx]

    # ---------------------------------- Internal ---------------------------------------------

    def __init__(self, stencil):
        dim = len(stencil[0])

        offsetSym = BoundaryOffsetInfo._offsetSymbols(dim)
        code = "\n"
        for i in range(dim):
            offsetStr = ", ".join([str(d[i]) for d in stencil])
            code += "const int64_t %s [] = { %s };\n" % (offsetSym[i].name, offsetStr)

        invDirs = []
        for direction in stencil:
            inverseDir = tuple([-i for i in direction])
            invDirs.append(str(stencil.index(inverseDir)))

        code += "const int %s [] = { %s };\n" % (self.INV_DIR_SYMBOL.name, ", ".join(invDirs))
        offsetSymbols = BoundaryOffsetInfo._offsetSymbols(dim)
        super(BoundaryOffsetInfo, self).__init__(code, symbolsRead=set(),
                                                 symbolsDefined=set(offsetSymbols + [self.INV_DIR_SYMBOL]))

    @staticmethod
    def _offsetSymbols(dim):
        return [TypedSymbol("c_%d" % (d,), createType(np.int64)) for d in range(dim)]

    INV_DIR_SYMBOL = TypedSymbol("invDir", "int")


def createBoundaryKernel(field, indexField, stencil, boundaryFunctor, target='cpu', openMP=True):
    elements = [BoundaryOffsetInfo(stencil)]
    indexArrDtype = indexField.dtype.numpyDtype
    dirSymbol = TypedSymbol("dir", indexArrDtype.fields['dir'][0])
    elements += [sp.Eq(dirSymbol, indexField[0]('dir'))]
    elements += boundaryFunctor(field, directionSymbol=dirSymbol, indexField=indexField)
    return createIndexedKernel(elements, [indexField], target=target, cpuOpenMP=openMP)
