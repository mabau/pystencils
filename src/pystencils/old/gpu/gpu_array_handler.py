try:
    import cupy as cp
    import cupyx as cpx
except ImportError:
    cp = None
    cpx = None

import numpy as np


class GPUArrayHandler:
    def __init__(self, device_number):
        self._device_number = device_number

    def zeros(self, shape, dtype=np.float64, order='C'):
        with cp.cuda.Device(self._device_number):
            return cp.zeros(shape=shape, dtype=dtype, order=order)

    def ones(self, shape, dtype=np.float64, order='C'):
        with cp.cuda.Device(self._device_number):
            return cp.ones(shape=shape, dtype=dtype, order=order)

    def empty(self, shape, dtype=np.float64, order='C'):
        with cp.cuda.Device(self._device_number):
            return cp.empty(shape=shape, dtype=dtype, order=order)

    def to_gpu(self, numpy_array):
        swaps = _get_index_swaps(numpy_array)
        if numpy_array.base is not None and isinstance(numpy_array.base, np.ndarray):
            with cp.cuda.Device(self._device_number):
                gpu_array = cp.asarray(numpy_array.base)
            for a, b in reversed(swaps):
                gpu_array = gpu_array.swapaxes(a, b)
            return gpu_array
        else:
            return cp.asarray(numpy_array)

    def upload(self, array, numpy_array):
        assert self._device_number == array.device.id
        if numpy_array.base is not None and isinstance(numpy_array.base, np.ndarray):
            with cp.cuda.Device(self._device_number):
                array.base.set(numpy_array.base)
        else:
            with cp.cuda.Device(self._device_number):
                array.set(numpy_array)

    def download(self, array, numpy_array):
        assert self._device_number == array.device.id
        if numpy_array.base is not None and isinstance(numpy_array.base, np.ndarray):
            with cp.cuda.Device(self._device_number):
                numpy_array.base[:] = array.base.get()
        else:
            with cp.cuda.Device(self._device_number):
                numpy_array[:] = array.get()

    def randn(self, shape, dtype=np.float64):
        with cp.cuda.Device(self._device_number):
            return cp.random.randn(*shape, dtype=dtype)

    @staticmethod
    def pinned_numpy_array(layout, shape, dtype):
        assert set(layout) == set(range(len(shape))), "Wrong layout descriptor"
        cur_layout = list(range(len(shape)))
        swaps = []
        for i in range(len(layout)):
            if cur_layout[i] != layout[i]:
                index_to_swap_with = cur_layout.index(layout[i])
                swaps.append((i, index_to_swap_with))
                cur_layout[i], cur_layout[index_to_swap_with] = cur_layout[index_to_swap_with], cur_layout[i]
        assert tuple(cur_layout) == tuple(layout)

        shape = list(shape)
        for a, b in swaps:
            shape[a], shape[b] = shape[b], shape[a]

        res = cpx.empty_pinned(tuple(shape), order='c', dtype=dtype)

        for a, b in reversed(swaps):
            res = res.swapaxes(a, b)
        return res

    from_numpy = to_gpu


class GPUNotAvailableHandler:
    def __getattribute__(self, name):
        raise NotImplementedError("Unable to utilise cupy! Please make sure cupy works correctly in your setup!")


def _get_index_swaps(array):
    swaps = []
    if array.base is not None and isinstance(array.base, np.ndarray):
        for stride in array.base.strides:
            index_base = array.base.strides.index(stride)
            index_view = array.strides.index(stride)
            if index_base != index_view and (index_view, index_base) not in swaps:
                swaps.append((index_base, index_view))
    return swaps
