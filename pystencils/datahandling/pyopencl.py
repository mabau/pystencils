try:
    import pyopencl.array as gpuarray
except ImportError:
    gpuarray = None

import numpy as np

import pystencils


class PyOpenClArrayHandler:

    def __init__(self, queue):
        if not queue:
            from pystencils.opencl.opencljit import get_global_cl_queue
            queue = get_global_cl_queue()
        assert queue, "OpenCL queue missing!\n" \
            "Use `import pystencils.opencl.autoinit` if you want it to be automatically created"
        self.queue = queue

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
            return gpuarray.empty(self.queue, shape, dtype)

    def to_gpu(self, array):
        return gpuarray.to_device(self.queue, array)

    def upload(self, gpuarray, numpy_array):
        gpuarray.set(numpy_array, self.queue)

    def download(self, gpuarray, numpy_array):
        gpuarray.get(self.queue, numpy_array)

    def randn(self, shape, dtype=np.float64):
        cpu_array = np.random.randn(*shape).astype(dtype)
        return self.from_numpy(cpu_array)

    from_numpy = to_gpu
