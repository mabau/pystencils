import sympy as sp
import numpy as np
from pystencils import Field
from pystencils.slicing import normalizeSlice, getPeriodicBoundarySrcDstSlices
from pystencils.gpucuda import makePythonFunction
from pystencils.gpucuda.kernelcreation import createCUDAKernel


def createCopyKernel(domainSize, fromSlice, toSlice, indexDimensions=0, indexDimShape=1, dtype=np.float64):
    """Copies a rectangular part of an array to another non-overlapping part"""
    if indexDimensions not in (0, 1):
        raise NotImplementedError("Works only for one or zero index coordinates")

    f = Field.createGeneric("pdfs", len(domainSize), indexDimensions=indexDimensions, dtype=dtype)
    normalizedFromSlice = normalizeSlice(fromSlice, f.spatialShape)
    normalizedToSlice = normalizeSlice(toSlice, f.spatialShape)

    offset = [s1.start - s2.start for s1, s2 in zip(normalizedFromSlice, normalizedToSlice)]
    assert offset == [s1.stop - s2.stop for s1, s2 in zip(normalizedFromSlice, normalizedToSlice)], "Slices have to have same size"

    updateEqs = []
    for i in range(indexDimShape):
        eq = sp.Eq(f(i), f[tuple(offset)](i))
        updateEqs.append(eq)

    ast = createCUDAKernel(updateEqs, iterationSlice=toSlice)
    return makePythonFunction(ast)


def getPeriodicBoundaryFunctor(stencil, domainSize, indexDimensions=0, indexDimShape=1, ghostLayers=1,
                               thickness=None, dtype=float):
    srcDstSliceTuples = getPeriodicBoundarySrcDstSlices(stencil, ghostLayers, thickness)
    kernels = []
    indexDimensions = indexDimensions
    for srcSlice, dstSlice in srcDstSliceTuples:
        kernels.append(createCopyKernel(domainSize, srcSlice, dstSlice, indexDimensions, indexDimShape, dtype))

    def functor(pdfs, **kwargs):
        for kernel in kernels:
            kernel(pdfs=pdfs)

    return functor
