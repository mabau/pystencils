import numpy as np
import itertools
import warnings

try:
    import pyximport;

    pyximport.install()
    from pystencils.boundaries.createindexlistcython import createBoundaryIndexList2D, createBoundaryIndexList3D

    cythonFuncsAvailable = True
except Exception:
    cythonFuncsAvailable = False
    createBoundaryIndexList2D = None
    createBoundaryIndexList3D = None

boundaryIndexArrayCoordinateNames = ["x", "y", "z"]
directionMemberName = "dir"


def numpyDataTypeForBoundaryObject(boundaryObject, dim):
    coordinateNames = boundaryIndexArrayCoordinateNames[:dim]
    return np.dtype([(name, np.int32) for name in coordinateNames] +
                    [(directionMemberName, np.int32)] +
                    [(i[0], i[1].numpyDtype) for i in boundaryObject.additionalData], align=True)


def _createBoundaryIndexListPython(flagFieldArr, nrOfGhostLayers, boundaryMask, fluidMask, stencil):
    coordinateNames = boundaryIndexArrayCoordinateNames[:len(flagFieldArr.shape)]
    indexArrDtype = np.dtype([(name, np.int32) for name in coordinateNames] + [(directionMemberName, np.int32)])

    result = []
    gl = nrOfGhostLayers
    for cell in itertools.product(*reversed([range(gl, i-gl) for i in flagFieldArr.shape])):
        cell = cell[::-1]
        if not flagFieldArr[cell] & fluidMask:
            continue
        for dirIdx, direction in enumerate(stencil):
            neighborCell = tuple([cell_i + dir_i for cell_i, dir_i in zip(cell, direction)])
            if flagFieldArr[neighborCell] & boundaryMask:
                result.append(cell + (dirIdx,))

    return np.array(result, dtype=indexArrDtype)


def createBoundaryIndexList(flagField, stencil, boundaryMask, fluidMask, nrOfGhostLayers=1):
    dim = len(flagField.shape)
    coordinateNames = boundaryIndexArrayCoordinateNames[:dim]
    indexArrDtype = np.dtype([(name, np.int32) for name in coordinateNames] + [(directionMemberName, np.int32)])

    if cythonFuncsAvailable:
        stencil = np.array(stencil, dtype=np.int32)
        if dim == 2:
            idxList = createBoundaryIndexList2D(flagField, nrOfGhostLayers, boundaryMask, fluidMask, stencil)
        elif dim == 3:
            idxList = createBoundaryIndexList3D(flagField, nrOfGhostLayers, boundaryMask, fluidMask, stencil)
        else:
            raise ValueError("Flag field has to be a 2 or 3 dimensional numpy array")
        return np.array(idxList, dtype=indexArrDtype)
    else:
        if flagField.size > 1e6:
            warnings.warn("Boundary setup may take very long! Consider installing cython to speed it up")
        return _createBoundaryIndexListPython(flagField, nrOfGhostLayers, boundaryMask, fluidMask, stencil)


def createBoundaryIndexArray(flagField, stencil, boundaryMask, fluidMask, boundaryObject, nrOfGhostLayers=1):
    idxArray = createBoundaryIndexList(flagField, stencil, boundaryMask, fluidMask, nrOfGhostLayers)
    dim = len(flagField.shape)

    if boundaryObject.additionalData:
        coordinateNames = boundaryIndexArrayCoordinateNames[:dim]
        indexArrDtype = numpyDataTypeForBoundaryObject(boundaryObject, dim)
        extendedIdxField = np.empty(len(idxArray), dtype=indexArrDtype)
        for prop in coordinateNames + ['dir']:
            extendedIdxField[prop] = idxArray[prop]

        idxArray = extendedIdxField

    return idxArray
