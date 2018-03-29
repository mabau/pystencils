from pyevtk.vtk import VtkFile, VtkImageData
from pyevtk.hl import _addDataToFile, _appendDataToFile


def imageToVTK(path, cellData, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)):
    """
    Writes numpy data to VTK

    Numpy arrays have to be contiguous in memory - if this is a problem call :func:`numpy.ascontiguousarray` first

    Patched version of same pyevtk function that also supports vector-valued data

    :param path: path with file name, without file ending (.vtk) where data should be stored
    :param cellData: dictionary, mapping name of the data to a 3D numpy array, or to a 3-tuple of 3D numpy arrays
                     in case of vector-valued data
    :param origin: 3-tuple describing the origin of the field in 3D
    :param spacing: 3-tuple describing the grid spacing in x,y, z direction
    :returns path to file that was written


    Example:

    >>> from tempfile import TemporaryDirectory
    >>> import os
    >>> import numpy as np
    >>> with TemporaryDirectory() as tmp_dir:
    ...     path = os.path.join(tmp_dir, 'out')
    ...     size = (20, 20, 20)
    ...     resFile = imageToVTK(path, cellData = {'someScalarField': np.zeros(size),
    ...                                            'someVectorField': (np.zeros(size), np.ones(size), np.zeros(size))
    ...                                           })
    """

    # Extract dimensions
    start = (0, 0, 0)
    end = None

    keys = list(cellData.keys())
    data = cellData[keys[0]]
    if hasattr(data, 'shape'):
        end = data.shape
    elif isinstance(data, tuple):
        shapes = set(d.shape for d in data)
        if len(shapes) > 1:
            raise ValueError("All components have to have the same shape")
        end = shapes.pop()

    # Write data to file
    w = VtkFile(path, VtkImageData)
    w.openGrid(start=start, end=end, origin=origin, spacing=spacing)
    w.openPiece(start=start, end=end)
    _addDataToFile(w, cellData, pointData=None)
    w.closePiece()
    w.closeGrid()
    _appendDataToFile(w, cellData, pointData=None)
    w.save()
    return w.getFileName()
