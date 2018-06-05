import numpy as np


def aligned_empty(shape, byte_alignment=32, dtype=np.float64, byte_offset=0, order='C', align_inner_coordinate=True):
    """
    Creates an aligned empty numpy array

    Args:
        shape: size of the array
        byte_alignment: alignment in bytes, for the start address of the array holds (a % byte_alignment) == 0
        dtype: numpy data type
        byte_offset: offset in bytes for position that should be aligned i.e. (a+byte_offset) % byte_alignment == 0
                    typically used to align first inner cell instead of ghost layer
        order: storage linearization order
        align_inner_coordinate: if True, the start of the innermost coordinate lines are aligned as well
    """
    if (not align_inner_coordinate) or (not hasattr(shape, '__len__')):
        size = np.prod(shape)
        d = np.dtype(dtype)
        # 2 * byte_alignment instead of 1 * byte_alignment to have slack in the end such that
        # vectorized loops can access vector_width elements further and don't require a tail loop
        tmp = np.empty(size * d.itemsize + 2 * byte_alignment, dtype=np.uint8)
        address = tmp.__array_interface__['data'][0]
        offset = (byte_alignment - (address + byte_offset) % byte_alignment) % byte_alignment
        return tmp[offset:offset + size * d.itemsize].view(dtype=d).reshape(shape, order=order)
    else:
        if order == 'C':
            dim0_size = shape[-1]
            dim0 = -1
            dim1_size = np.prod(shape[:-1])
        else:
            dim0_size = shape[0]
            dim0 = 0
            dim1_size = np.prod(shape[1:])
        d = np.dtype(dtype)

        assert byte_alignment >= d.itemsize and byte_alignment % d.itemsize == 0
        padding = (byte_alignment - ((dim0_size * d.itemsize) % byte_alignment)) % byte_alignment

        size = dim1_size * padding + np.prod(shape) * d.itemsize
        tmp = aligned_empty(size, byte_alignment=byte_alignment, dtype=np.uint8, byte_offset=byte_offset)
        tmp = tmp.view(dtype=dtype)
        shape_in_bytes = [i for i in shape]
        shape_in_bytes[dim0] = dim0_size + padding // d.itemsize
        tmp = tmp.reshape(shape_in_bytes, order=order)
        if tmp.flags['C_CONTIGUOUS']:
            tmp = tmp[..., :shape[-1]]
        else:
            tmp = tmp[:shape[0], ...]

        return tmp


def aligned_zeros(shape, byte_alignment=16, dtype=float, byte_offset=0, order='C', align_inner_coordinate=True):
    arr = aligned_empty(shape, dtype=dtype, byte_offset=byte_offset,
                        order=order, byte_alignment=byte_alignment, align_inner_coordinate=align_inner_coordinate)
    x = np.zeros((), arr.dtype)
    arr[...] = x
    return arr


def aligned_ones(shape, byte_alignment=16, dtype=float, byte_offset=0, order='C', align_inner_coordinate=True):
    arr = aligned_empty(shape, dtype=dtype, byte_offset=byte_offset,
                        order=order, byte_alignment=byte_alignment, align_inner_coordinate=align_inner_coordinate)
    x = np.ones((), arr.dtype)
    arr[...] = x
    return arr
