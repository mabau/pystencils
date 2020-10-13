import numpy as np

from pystencils.backends.cbackend import get_headers
from pystencils.backends.opencl_backend import generate_opencl
from pystencils.gpucuda.cudajit import _build_numpy_argument_list, _check_arguments
from pystencils.include import get_pystencils_include_path
from pystencils.kernel_wrapper import KernelWrapper

USE_FAST_MATH = True


_global_cl_ctx = None
_global_cl_queue = None


def get_global_cl_queue():
    return _global_cl_queue


def get_global_cl_ctx():
    return _global_cl_ctx


def init_globally(device_index=0):
    import pyopencl as cl
    global _global_cl_ctx
    global _global_cl_queue
    _global_cl_ctx = cl.create_some_context(device_index)
    _global_cl_queue = cl.CommandQueue(_global_cl_ctx)


def init_globally_with_context(opencl_ctx, opencl_queue):
    global _global_cl_ctx
    global _global_cl_queue
    _global_cl_ctx = opencl_ctx
    _global_cl_queue = opencl_queue


def clear_global_ctx():
    global _global_cl_ctx
    global _global_cl_queue
    _global_cl_ctx = None
    _global_cl_queue = None


def make_python_function(kernel_function_node, opencl_queue, opencl_ctx, argument_dict=None, custom_backend=None):
    """
    Creates a **OpenCL** kernel function from an abstract syntax tree which
    was created for the ``target='gpu'`` e.g. by :func:`pystencils.gpucuda.create_cuda_kernel`
    or :func:`pystencils.gpucuda.created_indexed_cuda_kernel`

    Args:
        opencl_queue: a valid :class:`pyopencl.CommandQueue`
        opencl_ctx: a valid :class:`pyopencl.Context`
        kernel_function_node: the abstract syntax tree
        argument_dict: parameters passed here are already fixed. Remaining parameters have to be passed to the
                       returned kernel functor.

    Returns:
        compiled kernel as Python function
    """
    import pyopencl as cl

    if not opencl_ctx:
        opencl_ctx = _global_cl_ctx
    if not opencl_queue:
        opencl_queue = _global_cl_queue

    assert opencl_ctx, "No valid OpenCL context!\n" \
        "Use `import pystencils.opencl.autoinit` if you want it to be automatically created"
    assert opencl_queue, "No valid OpenCL queue!\n" \
        "Use `import pystencils.opencl.autoinit` if you want it to be automatically created"

    if argument_dict is None:
        argument_dict = {}

    # check if double precision is supported and required
    if any([d.double_fp_config == 0 for d in opencl_ctx.devices]):
        for param in kernel_function_node.get_parameters():
            if param.symbol.dtype.base_type:
                if param.symbol.dtype.base_type.numpy_dtype == np.float64:
                    raise ValueError('OpenCL device does not support double precision')
            else:
                if param.symbol.dtype.numpy_dtype == np.float64:
                    raise ValueError('OpenCL device does not support double precision')

    # Changing of kernel name necessary since compilation with default name "kernel" is not possible (OpenCL keyword!)
    kernel_function_node.function_name = "opencl_" + kernel_function_node.function_name
    header_list = ['"opencl_stdint.h"'] + list(get_headers(kernel_function_node))
    includes = "\n".join(["#include %s" % (include_file,) for include_file in header_list])

    code = includes + "\n"
    code += "#define FUNC_PREFIX __kernel\n"
    code += "#define RESTRICT restrict\n\n"
    code += str(generate_opencl(kernel_function_node, custom_backend=custom_backend))
    options = []
    if USE_FAST_MATH:
        options.append("-cl-unsafe-math-optimizations")
        options.append("-cl-mad-enable")
        options.append("-cl-fast-relaxed-math")
        options.append("-cl-finite-math-only")
    options.append("-I")
    options.append(get_pystencils_include_path())
    mod = cl.Program(opencl_ctx, code).build(options=options)
    func = getattr(mod, kernel_function_node.function_name)

    parameters = kernel_function_node.get_parameters()

    cache = {}
    cache_values = []

    def wrapper(**kwargs):
        key = hash(tuple((k, v.ctypes.data, v.strides, v.shape) if isinstance(v, np.ndarray) else (k, id(v))
                         for k, v in kwargs.items()))
        try:
            args, block_and_thread_numbers = cache[key]
            func(opencl_queue, block_and_thread_numbers['grid'], block_and_thread_numbers['block'], *args)
        except KeyError:
            full_arguments = argument_dict.copy()
            full_arguments.update(kwargs)
            assert not any(isinstance(a, np.ndarray)
                           for a in full_arguments.values()), 'Calling a OpenCL kernel with a Numpy array!'
            assert not any('pycuda' in str(type(a))
                           for a in full_arguments.values()), 'Calling a OpenCL kernel with a PyCUDA array!'
            shape = _check_arguments(parameters, full_arguments)

            indexing = kernel_function_node.indexing
            block_and_thread_numbers = indexing.call_parameters(shape)
            block_and_thread_numbers['block'] = tuple(int(i) for i in block_and_thread_numbers['block'])
            block_and_thread_numbers['grid'] = tuple(int(b * g) for (b, g) in zip(block_and_thread_numbers['block'],
                                                                                  block_and_thread_numbers['grid']))

            args = _build_numpy_argument_list(parameters, full_arguments)
            args = [a.data if hasattr(a, 'data') else a for a in args]
            cache[key] = (args, block_and_thread_numbers)
            cache_values.append(kwargs)  # keep objects alive such that ids remain unique
            func(opencl_queue, block_and_thread_numbers['grid'], block_and_thread_numbers['block'], *args)

    wrapper.ast = kernel_function_node
    wrapper.parameters = kernel_function_node.get_parameters()
    wrapper = KernelWrapper(wrapper, parameters, kernel_function_node)
    return wrapper
