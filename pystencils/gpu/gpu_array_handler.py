try:
    import cupy as cp
except ImportError:
    cp = None

import numpy as np
import pystencils


class GPUArrayHandler:
    @staticmethod
    def zeros(shape, dtype=np.float64, order='C'):
        with cp.cuda.Device(pystencils.GPU_DEVICE):
            return cp.zeros(shape=shape, dtype=dtype, order=order)

    @staticmethod
    def ones(shape, dtype=np.float64, order='C'):
        with cp.cuda.Device(pystencils.GPU_DEVICE):
            return cp.ones(shape=shape, dtype=dtype, order=order)

    @staticmethod
    def empty(shape, dtype=np.float64, order='C'):
        with cp.cuda.Device(pystencils.GPU_DEVICE):
            return cp.empty(shape=shape, dtype=dtype, order=order)

    @staticmethod
    def to_gpu(numpy_array):
        with cp.cuda.Device(pystencils.GPU_DEVICE):
            return cp.asarray(numpy_array)

    @staticmethod
    def upload(array, numpy_array):
        with cp.cuda.Device(pystencils.GPU_DEVICE):
            array.set(numpy_array)

    @staticmethod
    def download(array, numpy_array):
        with cp.cuda.Device(pystencils.GPU_DEVICE):
            numpy_array[:] = array.get()

    @staticmethod
    def randn(shape, dtype=np.float64):
        with cp.cuda.Device(pystencils.GPU_DEVICE):
            return cp.random.randn(*shape, dtype=dtype)

    from_numpy = to_gpu


class GPUNotAvailableHandler:
    def __getattribute__(self, name):
        raise NotImplementedError("Unable to utilise cupy! Please make sure cupy works correctly in your setup!")
