import numpy as np


def aligned_empty(shape, byte_alignment=True, dtype=np.float64, byte_offset=0, order='C', align_inner_coordinate=True):
    """
    Creates an aligned empty numpy array

    Args:
        shape: size of the array
        byte_alignment: alignment in bytes, for the start address of the array holds (a % byte_alignment) == 0
                        By default, use the maximum required by the CPU (or 512 bits if this cannot be detected).
                        When 'cacheline' is specified, the size of a cache line is used.
        dtype: numpy data type
        byte_offset: offset in bytes for position that should be aligned i.e. (a+byte_offset) % byte_alignment == 0
                    typically used to align first inner cell instead of ghost layer
        order: storage linearization order
        align_inner_coordinate: if True, the start of the innermost coordinate lines are aligned as well
    """
    if byte_alignment is True or byte_alignment == 'cacheline':
        from pystencils.backends.simd_instruction_sets import (get_supported_instruction_sets, get_cacheline_size,
                                                               get_vector_instruction_set)

        instruction_sets = get_supported_instruction_sets()
        if instruction_sets is None:
            byte_alignment = 64
        elif byte_alignment == 'cacheline':
            cacheline_sizes = [get_cacheline_size(is_name) for is_name in instruction_sets]
            if all([s is None for s in cacheline_sizes]):
                widths = [get_vector_instruction_set(dtype, is_name)['width'] * np.dtype(dtype).itemsize
                          for is_name in instruction_sets
                          if type(get_vector_instruction_set(dtype, is_name)['width']) is int]
                byte_alignment = 64 if all([s is None for s in widths]) else max(widths)
            else:
                byte_alignment = max([s for s in cacheline_sizes if s is not None])
        elif not any([type(get_vector_instruction_set(dtype, is_name)['width']) is int
                      for is_name in instruction_sets]):
            byte_alignment = 64
        else:
            byte_alignment = max([get_vector_instruction_set(dtype, is_name)['width'] * np.dtype(dtype).itemsize
                                  for is_name in instruction_sets
                                  if type(get_vector_instruction_set(dtype, is_name)['width']) is int])
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


def aligned_zeros(shape, byte_alignment=True, dtype=np.float64, byte_offset=0, order='C', align_inner_coordinate=True):
    arr = aligned_empty(shape, dtype=dtype, byte_offset=byte_offset,
                        order=order, byte_alignment=byte_alignment, align_inner_coordinate=align_inner_coordinate)
    x = np.zeros((), arr.dtype)
    arr[...] = x
    return arr


def aligned_ones(shape, byte_alignment=True, dtype=np.float64, byte_offset=0, order='C', align_inner_coordinate=True):
    arr = aligned_empty(shape, dtype=dtype, byte_offset=byte_offset,
                        order=order, byte_alignment=byte_alignment, align_inner_coordinate=align_inner_coordinate)
    x = np.ones((), arr.dtype)
    arr[...] = x
    return arr
