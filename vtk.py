from pyevtk.vtk import VtkFile, VtkImageData
from pyevtk.hl import _addDataToFile, _appendDataToFile


def imageToVTK(path, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0), cellData=None, pointData=None):
    """Patched version of same pyevtk function that also support vector data"""
    assert cellData != None or pointData != None
    # Extract dimensions
    start = (0, 0, 0)
    end = None
    if cellData:
        keys = list(cellData.keys())
        data = cellData[keys[0]]
        if hasattr(data, 'shape'):
            end = data.shape
        elif data[0].ndim == 3 and data[1].ndim == 3 and data[0].ndim == 3:
            keys = list(cellData.keys())
            data = cellData[keys[0]]
            end = data[0].shape
    elif pointData:
        keys = list(pointData.keys())
        data = pointData[keys[0]]
        if hasattr(data, 'shape'):
            end = data.shape
            end = (end[0] - 1, end[1] - 1, end[2] - 1)
        # Added for vector support...
        elif data[0].ndim == 3 and data[1].ndim == 3 and data[0].ndim == 3:
            keys = list(pointData.keys())
            data = pointData[keys[0]]
            end = data[0].shape
            end = (end[0] - 1, end[1] - 1, end[2] - 1)
    # Write data to file
    w = VtkFile(path, VtkImageData)
    w.openGrid(start=start, end=end, origin=origin, spacing=spacing)
    w.openPiece(start=start, end=end)
    _addDataToFile(w, cellData, pointData)
    w.closePiece()
    w.closeGrid()
    _appendDataToFile(w, cellData, pointData)
    w.save()
    return w.getFileName()
