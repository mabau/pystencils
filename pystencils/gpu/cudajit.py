import numpy as np

import pystencils
from pystencils.backends.cbackend import get_headers
from pystencils.backends.cuda_backend import generate_cuda
from pystencils.typing import StructType
from pystencils.field import FieldType
from pystencils.include import get_pystencils_include_path
from pystencils.kernel_wrapper import KernelWrapper
from pystencils.typing.typed_sympy import FieldPointerSymbol

USE_FAST_MATH = True


def get_cubic_interpolation_include_paths():
    from os.path import join, dirname

    return [join(dirname(__file__), "CubicInterpolationCUDA", "code"),
            join(dirname(__file__), "CubicInterpolationCUDA", "code", "internal")]


def make_python_function(kernel_function_node, argument_dict=None, custom_backend=None):
    """
    Creates a kernel function from an abstract syntax tree which
    was created e.g. by :func:`pystencils.gpu.create_cuda_kernel`
    or :func:`pystencils.gpu.created_indexed_cuda_kernel`

    Args:
        kernel_function_node: the abstract syntax tree
        argument_dict: parameters passed here are already fixed. Remaining parameters have to be passed to the
                       returned kernel functor.
        custom_backend: use own custom printer for code generation

    Returns:
        compiled kernel as Python function
    """
    import cupy as cp

    if argument_dict is None:
        argument_dict = {}

    if cp.cuda.runtime.is_hip:
        header_list = ['"gpu_defines.h"'] + list(get_headers(kernel_function_node))
    else:
        header_list = ['"gpu_defines.h"', '<cstdint>'] + list(get_headers(kernel_function_node))
    includes = "\n".join([f"#include {include_file}" for include_file in header_list])

    code = includes + "\n"
    code += "#define FUNC_PREFIX __global__\n"
    code += "#define RESTRICT __restrict__\n\n"
    code += str(generate_cuda(kernel_function_node, custom_backend=custom_backend))
    code = 'extern "C" {\n%s\n}\n' % code

    options = ["-w", "-std=c++11"]
    if USE_FAST_MATH:
        options.append("-use_fast_math")
    options.append("-I" + get_pystencils_include_path())

    mod = cp.RawModule(code=code, options=tuple(options), backend="nvrtc", jitify=True)
    func = mod.get_function(kernel_function_node.function_name)

    parameters = kernel_function_node.get_parameters()

    cache = {}
    cache_values = []

    def wrapper(**kwargs):
        key = hash(tuple((k, v.ctypes.data, v.strides, v.shape) if isinstance(v, np.ndarray) else (k, id(v))
                         for k, v in kwargs.items()))
        try:
            args, block_and_thread_numbers = cache[key]
            with cp.cuda.Device(pystencils.GPU_DEVICE):
                func(block_and_thread_numbers['grid'], block_and_thread_numbers['block'], args)
        except KeyError:
            full_arguments = argument_dict.copy()
            full_arguments.update(kwargs)
            shape = _check_arguments(parameters, full_arguments)

            indexing = kernel_function_node.indexing
            block_and_thread_numbers = indexing.call_parameters(shape)
            block_and_thread_numbers['block'] = tuple(int(i) for i in block_and_thread_numbers['block'])
            block_and_thread_numbers['grid'] = tuple(int(i) for i in block_and_thread_numbers['grid'])

            args = tuple(_build_numpy_argument_list(parameters, full_arguments))
            cache[key] = (args, block_and_thread_numbers)
            cache_values.append(kwargs)  # keep objects alive such that ids remain unique
            with cp.cuda.Device(pystencils.GPU_DEVICE):
                func(block_and_thread_numbers['grid'], block_and_thread_numbers['block'], args)
            # useful for debugging:
            # with cp.cuda.Device(pystencils.GPU_DEVICE):
            #     cp.cuda.runtime.deviceSynchronize()

        # cuda.Context.synchronize() # useful for debugging, to get errors right after kernel was called
    ast = kernel_function_node
    parameters = kernel_function_node.get_parameters()
    wrapper = KernelWrapper(wrapper, parameters, ast)
    wrapper.num_regs = func.num_regs
    return wrapper


def _build_numpy_argument_list(parameters, argument_dict):
    argument_dict = {k: v for k, v in argument_dict.items()}
    result = []

    for param in parameters:
        if param.is_field_pointer:
            array = argument_dict[param.field_name]
            actual_type = array.dtype
            expected_type = param.fields[0].dtype.numpy_dtype
            if expected_type != actual_type:
                raise ValueError(f"Data type mismatch for field {param.field_name}. "
                                 f"Expected {expected_type} got {actual_type}.")
            result.append(array)
        elif param.is_field_stride:
            cast_to_dtype = param.symbol.dtype.numpy_dtype.type
            array = argument_dict[param.field_name]
            stride = cast_to_dtype(array.strides[param.symbol.coordinate] // array.dtype.itemsize)
            result.append(stride)
        elif param.is_field_shape:
            cast_to_dtype = param.symbol.dtype.numpy_dtype.type
            array = argument_dict[param.field_name]
            result.append(cast_to_dtype(array.shape[param.symbol.coordinate]))
        else:
            expected_type = param.symbol.dtype.numpy_dtype
            result.append(expected_type.type(argument_dict[param.symbol.name]))

    assert len(result) == len(parameters)
    return result


def _check_arguments(parameter_specification, argument_dict):
    """
    Checks if parameters passed to kernel match the description in the AST function node.
    If not it raises a ValueError, on success it returns the array shape that determines the CUDA blocks and threads
    """
    argument_dict = {k: v for k, v in argument_dict.items()}
    array_shapes = set()
    index_arr_shapes = set()

    for param in parameter_specification:
        if isinstance(param.symbol, FieldPointerSymbol):
            symbolic_field = param.fields[0]

            try:
                field_arr = argument_dict[symbolic_field.name]
            except KeyError:
                raise KeyError(f"Missing field parameter for kernel call {str(symbolic_field)}")

            if symbolic_field.has_fixed_shape:
                symbolic_field_shape = tuple(int(i) for i in symbolic_field.shape)
                if isinstance(symbolic_field.dtype, StructType):
                    symbolic_field_shape = symbolic_field_shape[:-1]
                if symbolic_field_shape != field_arr.shape:
                    raise ValueError(f"Passed array {symbolic_field.name} has shape {str(field_arr.shape)} "
                                     f"which does not match expected shape {str(symbolic_field.shape)}")
            if symbolic_field.has_fixed_shape:
                symbolic_field_strides = tuple(int(i) * field_arr.dtype.itemsize for i in symbolic_field.strides)
                if isinstance(symbolic_field.dtype, StructType):
                    symbolic_field_strides = symbolic_field_strides[:-1]
                if symbolic_field_strides != field_arr.strides:
                    raise ValueError(f"Passed array {symbolic_field.name} has strides {str(field_arr.strides)} "
                                     f"which does not match expected strides {str(symbolic_field_strides)}")

            if FieldType.is_indexed(symbolic_field):
                index_arr_shapes.add(field_arr.shape[:symbolic_field.spatial_dimensions])
            elif FieldType.is_generic(symbolic_field):
                array_shapes.add(field_arr.shape[:symbolic_field.spatial_dimensions])

    if len(array_shapes) > 1:
        raise ValueError(f"All passed arrays have to have the same size {str(array_shapes)}")
    if len(index_arr_shapes) > 1:
        raise ValueError(f"All passed index arrays have to have the same size {str(array_shapes)}")

    if len(index_arr_shapes) > 0:
        return list(index_arr_shapes)[0]
    else:
        return list(array_shapes)[0]
