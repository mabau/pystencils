import numpy as np
from pystencils.backends.cbackend import generate_c
from pystencils.transformations import symbol_name_to_variable_name
from pystencils.data_types import StructType, get_base_type
from pystencils.field import FieldType


def make_python_function(kernel_function_node, argument_dict=None):
    """
    Creates a kernel function from an abstract syntax tree which
    was created e.g. by :func:`pystencils.gpucuda.create_cuda_kernel`
    or :func:`pystencils.gpucuda.created_indexed_cuda_kernel`

    Args:
        kernel_function_node: the abstract syntax tree
        argument_dict: parameters passed here are already fixed. Remaining parameters have to be passed to the
                       returned kernel functor.

    Returns:
        compiled kernel as Python function
    """
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

    if argument_dict is None:
        argument_dict = {}

    code = "#include <cstdint>\n"
    code += "#define FUNC_PREFIX __global__\n"
    code += "#define RESTRICT __restrict__\n\n"
    code += str(generate_c(kernel_function_node))

    mod = SourceModule(code, options=["-w", "-std=c++11"])
    func = mod.get_function(kernel_function_node.function_name)

    parameters = kernel_function_node.parameters

    cache = {}
    cache_values = []

    def wrapper(**kwargs):
        key = hash(tuple((k, v.ctypes.data, v.strides, v.shape) if isinstance(v, np.ndarray) else (k, id(v))
                         for k, v in kwargs.items()))
        try:
            args, block_and_thread_numbers = cache[key]
            func(*args, **block_and_thread_numbers)
        except KeyError:
            full_arguments = argument_dict.copy()
            full_arguments.update(kwargs)
            shape = _check_arguments(parameters, full_arguments)

            indexing = kernel_function_node.indexing
            block_and_thread_numbers = indexing.call_parameters(shape)
            block_and_thread_numbers['block'] = tuple(int(i) for i in block_and_thread_numbers['block'])
            block_and_thread_numbers['grid'] = tuple(int(i) for i in block_and_thread_numbers['grid'])

            args = _build_numpy_argument_list(parameters, full_arguments)
            cache[key] = (args, block_and_thread_numbers)
            cache_values.append(kwargs)  # keep objects alive such that ids remain unique
            func(*args, **block_and_thread_numbers)
        #cuda.Context.synchronize() # useful for debugging, to get errors right after kernel was called
    wrapper.ast = kernel_function_node
    wrapper.parameters = kernel_function_node.parameters
    return wrapper


def _build_numpy_argument_list(parameters, argument_dict):
    import pycuda.driver as cuda

    argument_dict = {symbol_name_to_variable_name(k): v for k, v in argument_dict.items()}
    result = []
    for arg in parameters:
        if arg.isFieldArgument:
            field = argument_dict[arg.field_name]
            if arg.isFieldPtrArgument:
                actual_type = field.dtype
                expected_type = arg.dtype.base_type.numpy_dtype
                if expected_type != actual_type:
                    raise ValueError("Data type mismatch for field '%s'. Expected '%s' got '%s'." %
                                     (arg.field_name, expected_type, actual_type))
                result.append(field)
            elif arg.isFieldStrideArgument:
                dtype = get_base_type(arg.dtype).numpy_dtype
                stride_arr = np.array(field.strides, dtype=dtype) // field.dtype.itemsize
                result.append(cuda.In(stride_arr))
            elif arg.isFieldShapeArgument:
                dtype = get_base_type(arg.dtype).numpy_dtype
                shape_arr = np.array(field.shape, dtype=dtype)
                result.append(cuda.In(shape_arr))
            else:
                assert False
        else:
            param = argument_dict[arg.name]
            expected_type = arg.dtype.numpy_dtype
            result.append(expected_type.type(param))
    assert len(result) == len(parameters)
    return result


def _check_arguments(parameter_specification, argument_dict):
    """
    Checks if parameters passed to kernel match the description in the AST function node.
    If not it raises a ValueError, on success it returns the array shape that determines the CUDA blocks and threads
    """
    argument_dict = {symbol_name_to_variable_name(k): v for k, v in argument_dict.items()}
    array_shapes = set()
    index_arr_shapes = set()
    for arg in parameter_specification:
        if arg.isFieldArgument:
            try:
                field_arr = argument_dict[arg.field_name]
            except KeyError:
                raise KeyError("Missing field parameter for kernel call " + arg.field_name)

            symbolic_field = arg.field
            if arg.isFieldPtrArgument:
                if symbolic_field.has_fixed_shape:
                    symbolic_field_shape = tuple(int(i) for i in symbolic_field.shape)
                    if isinstance(symbolic_field.dtype, StructType):
                        symbolic_field_shape = symbolic_field_shape[:-1]
                    if symbolic_field_shape != field_arr.shape:
                        raise ValueError("Passed array '%s' has shape %s which does not match expected shape %s" %
                                         (arg.field_name, str(field_arr.shape), str(symbolic_field.shape)))
                if symbolic_field.has_fixed_shape:
                    symbolic_field_strides = tuple(int(i) * field_arr.dtype.itemsize for i in symbolic_field.strides)
                    if isinstance(symbolic_field.dtype, StructType):
                        symbolic_field_strides = symbolic_field_strides[:-1]
                    if symbolic_field_strides != field_arr.strides:
                        raise ValueError("Passed array '%s' has strides %s which does not match expected strides %s" %
                                         (arg.field_name, str(field_arr.strides), str(symbolic_field_strides)))

                if FieldType.is_indexed(symbolic_field):
                    index_arr_shapes.add(field_arr.shape[:symbolic_field.spatial_dimensions])
                elif not FieldType.is_buffer(symbolic_field):
                    array_shapes.add(field_arr.shape[:symbolic_field.spatial_dimensions])

    if len(array_shapes) > 1:
        raise ValueError("All passed arrays have to have the same size " + str(array_shapes))
    if len(index_arr_shapes) > 1:
        raise ValueError("All passed index arrays have to have the same size " + str(array_shapes))

    if len(index_arr_shapes) > 0:
        return list(index_arr_shapes)[0]
    else:
        return list(array_shapes)[0]



