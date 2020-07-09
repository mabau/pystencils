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
        return gpuarray.zeros(shape, dtype, order)

    def ones(self, shape, dtype, order='C'):
        return gpuarray.ones(shape, dtype, order)

    def empty(self, shape, dtype=np.float64, layout=None):
        if layout:
            cpu_array = pystencils.field.create_numpy_array_with_layout(shape, dtype, layout)
            return self.to_gpu(cpu_array)
        else:
            return gpuarray.empty(shape, dtype)

    def to_gpu(self, array):
        return gpuarray.to_gpu(array)

    def upload(self, gpuarray, numpy_array):
        gpuarray.set(numpy_array)

    def download(self, gpuarray, numpy_array):
        gpuarray.get(numpy_array)

    def randn(self, shape, dtype=np.float64):
        cpu_array = np.random.randn(*shape).astype(dtype)
        return self.to_gpu(cpu_array)

    from_numpy = to_gpu


class PyCudaNotAvailableHandler:
    def __getattribute__(self, name):
        raise NotImplementedError("Unable to initiaize PyCuda! "
                                  "Try to run `import pycuda.autoinit` to check whether PyCuda is working correctly!")
