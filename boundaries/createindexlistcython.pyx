# Workaround for cython bug
# see https://stackoverflow.com/questions/8024805/cython-compiled-c-extension-importerror-dynamic-module-does-not-define-init-fu
WORKAROUND = "Something"

import cython

ctypedef fused IntegerType:
    short
    int
    long
    long long
    unsigned short
    unsigned int
    unsigned long

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def createBoundaryIndexList2D(object[IntegerType, ndim=2] flagField,
                              int nrOfGhostLayers, IntegerType boundaryMask, IntegerType fluidMask,
                              object[int, ndim=2] stencil):
    cdef int xs, ys, x, y
    cdef int dirIdx, numDirections, dx, dy

    xs, ys = flagField.shape
    boundaryIndexList = []
    numDirections = stencil.shape[0]

    for y in range(nrOfGhostLayers,ys-nrOfGhostLayers):
        for x in range(nrOfGhostLayers,xs-nrOfGhostLayers):
            if flagField[x,y] & fluidMask:
                for dirIdx in range(1, numDirections):
                    dx = stencil[dirIdx,0]
                    dy = stencil[dirIdx,1]
                    if flagField[x+dx, y+dy] & boundaryMask:
                        boundaryIndexList.append((x,y, dirIdx))
    return boundaryIndexList


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def createBoundaryIndexList3D(object[IntegerType, ndim=3] flagField,
                              int nrOfGhostLayers, IntegerType boundaryMask, IntegerType fluidMask,
                              object[int, ndim=2] stencil):
    cdef int xs, ys, zs, x, y, z
    cdef int dirIdx, numDirections, dx, dy, dz

    xs, ys, zs = flagField.shape
    boundaryIndexList = []
    numDirections = stencil.shape[0]

    for z in range(nrOfGhostLayers, zs-nrOfGhostLayers):
        for y in range(nrOfGhostLayers,ys-nrOfGhostLayers):
            for x in range(nrOfGhostLayers,xs-nrOfGhostLayers):
                if flagField[x, y, z] & fluidMask:
                    for dirIdx in range(1, numDirections):
                        dx = stencil[dirIdx,0]
                        dy = stencil[dirIdx,1]
                        dz = stencil[dirIdx,2]
                        if flagField[x + dx, y + dy, z + dz] & boundaryMask:
                            boundaryIndexList.append((x,y,z, dirIdx))
    return boundaryIndexList


