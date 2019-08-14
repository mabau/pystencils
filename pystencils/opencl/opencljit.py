import numpy as np

from pystencils.backends.cbackend import generate_c, get_headers
from pystencils.gpucuda.cudajit import _build_numpy_argument_list, _check_arguments
from pystencils.include import get_pystencils_include_path

USE_FAST_MATH = True


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
    assert opencl_ctx, "No valid OpenCL context"
    assert opencl_queue, "No valid OpenCL queue"

    if argument_dict is None:
        argument_dict = {}

    kernel_function_node.function_name = "opencl_" + kernel_function_node.function_name
    header_list = ['"opencl_stdint.h"'] + list(get_headers(kernel_function_node))
    includes = "\n".join(["#include %s" % (include_file,) for include_file in header_list])

    code = includes + "\n"
    code += "#define FUNC_PREFIX __kernel\n"
    code += "#define RESTRICT restrict\n\n"
    code += str(generate_c(kernel_function_node, dialect='opencl', custom_backend=custom_backend))
    options = []
    if USE_FAST_MATH:
        options.append("-cl-unsafe-math-optimizations -cl-mad-enable -cl-fast-relaxed-math -cl-finite-math-only")
    options.append("-I \"" + get_pystencils_include_path() + "\"")
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
    return wrapper
