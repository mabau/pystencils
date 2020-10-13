try:
    import pycuda.gpuarray as gpuarray
except ImportError:
    gpuarray = None
import numpy as np

import pystencils


class PyCudaArrayHandler:

    def __init__(self):
        import pycuda.autoinit  # NOQA

    def zeros(self, shape, dtype=np.float64, order='C'):
        cpu_array = np.zeros(shape=shape, dtype=dtype, order=order)
        return self.to_gpu(cpu_array)

    def ones(self, shape, dtype=np.float64, order='C'):
        cpu_array = np.ones(shape=shape, dtype=dtype, order=order)
        return self.to_gpu(cpu_array)

    def empty(self, shape, dtype=np.float64, layout=None):
        if layout:
            cpu_array = pystencils.field.create_numpy_array_with_layout(shape=shape, dtype=dtype, layout=layout)
            return self.to_gpu(cpu_array)
        else:
            return gpuarray.empty(shape, dtype)

    @staticmethod
    def to_gpu(array):
        return gpuarray.to_gpu(array)

    @staticmethod
    def upload(array, numpy_array):
        array.set(numpy_array)

    @staticmethod
    def download(array, numpy_array):
        array.get(numpy_array)

    def randn(self, shape, dtype=np.float64):
        cpu_array = np.random.randn(*shape).astype(dtype)
        return self.to_gpu(cpu_array)

    from_numpy = to_gpu


class PyCudaNotAvailableHandler:
    def __getattribute__(self, name):
        raise NotImplementedError("Unable to initiaize PyCuda! "
                                  "Try to run `import pycuda.autoinit` to check whether PyCuda is working correctly!")
