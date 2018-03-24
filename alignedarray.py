import numpy as np


def aligned_empty(shape, byteAlignment=32, dtype=np.float64, byteOffset=0, order='C', alignInnerCoordinate=True):
    """
    Creates an aligned empty numpy array
    :param shape: size of the array
    :param byteAlignment: alignment in bytes, for the start address of the array holds (a % byteAlignment) == 0
    :param dtype: numpy data type
    :param byteOffset: offset in bytes for position that should be aligned i.e. (a+byteOffset) % byteAlignment == 0
                       typically used to align first inner cell instead of ghost layer
    :param order: storage linearization order
    :param alignInnerCoordinate: if True, the start of the innermost coordinate lines are aligned as well
    :return:
    """
    if (not alignInnerCoordinate) or (not hasattr(shape, '__len__')):
        N = np.prod(shape)
        d = np.dtype(dtype)
        tmp = np.empty(N * d.itemsize + byteAlignment, dtype=np.uint8)
        address = tmp.__array_interface__['data'][0]
        offset = (byteAlignment - (address + byteOffset) % byteAlignment) % byteAlignment
        return tmp[offset:offset + N * d.itemsize].view(dtype=d).reshape(shape, order=order)
    else:
        if order == 'C':
            ndim0 = shape[-1]
            dim0 = -1
            ndim1 = shape[-2]
        else:
            ndim0 = shape[0]
            dim0 = 0
            ndim1 = shape[1]
        d = np.dtype(dtype)

        assert byteAlignment >= d.itemsize and byteAlignment % d.itemsize == 0
        padding = (byteAlignment - ((ndim0 * d.itemsize) % byteAlignment)) % byteAlignment

        N = ndim1 * padding + np.prod(shape) * d.itemsize
        tmp = aligned_empty(N, byteAlignment=byteAlignment, dtype=np.uint8, byteOffset=byteOffset).view(dtype=dtype)
        bshape = [i for i in shape]
        bshape[dim0] = ndim0 + padding // d.itemsize
        tmp = tmp.reshape(bshape, order=order)
        if tmp.flags['C_CONTIGUOUS']:
            tmp = tmp[..., :shape[-1]]
        else:
            tmp = tmp[:shape[0], ...]

        return tmp


def aligned_zeros(shape, byteAlignment=16, dtype=float, byteOffset=0, order='C', alignInnerCoordinate=True):
    arr = aligned_empty(shape, dtype=dtype, byteOffset=byteOffset,
                        order=order, byteAlignment=byteAlignment, alignInnerCoordinate=alignInnerCoordinate)
    x = np.zeros((), arr.dtype)
    arr[...] = x
    return arr


def aligned_ones(shape, byteAlignment=16, dtype=float, byteOffset=0, order='C', alignInnerCoordinate=True):
    arr = aligned_empty(shape, dtype=dtype, byteOffset=byteOffset,
                        order=order, byteAlignment=byteAlignment, alignInnerCoordinate=alignInnerCoordinate)
    x = np.ones((), arr.dtype)
    arr[...] = x
    return arr
