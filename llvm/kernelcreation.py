from pystencils.transformations import insert_casts
from functools import partial
from pystencils.llvm.llvmjit import make_python_function


def create_kernel(assignments, function_name="kernel", type_info=None, split_groups=(),
                  iteration_slice=None, ghost_layers=None):
    """
    Creates an abstract syntax tree for a kernel function, by taking a list of update rules.

    Loops are created according to the field accesses in the equations.

    :param assignments: list of sympy equations, containing accesses to :class:`pystencils.field.Field`.
           Defining the update rules of the kernel
    :param function_name: name of the generated function - only important if generated code is written out
    :param type_info: a map from symbol name to a C type specifier. If not specified all symbols are assumed to
           be of type 'double' except symbols which occur on the left hand side of equations where the
           right hand side is a sympy Boolean which are assumed to be 'bool' .
    :param split_groups: Specification on how to split up inner loop into multiple loops. For details see
           transformation :func:`pystencils.transformation.split_inner_loop`
    :param iteration_slice: if not None, iteration is done only over this slice of the field
    :param ghost_layers: a sequence of pairs for each coordinate with lower and upper nr of ghost layers
                        if None, the number of ghost layers is determined automatically and assumed to be equal for a
                        all dimensions

    :return: :class:`pystencils.ast.KernelFunction` node
    """
    from pystencils.cpu import create_kernel
    code = create_kernel(assignments, function_name, type_info, split_groups, iteration_slice, ghost_layers)
    code = insert_casts(code)
    code.compile = partial(make_python_function, code)
    return code


def create_indexed_kernel(assignments, index_fields, function_name="kernel", type_info=None,
                          coordinate_names=('x', 'y', 'z')):
    """
    Similar to :func:`create_kernel`, but here not all cells of a field are updated but only cells with
    coordinates which are stored in an index field. This traversal method can e.g. be used for boundary handling.

    The coordinates are stored in a separated index_field, which is a one dimensional array with struct data type.
    This struct has to contain fields named 'x', 'y' and for 3D fields ('z'). These names are configurable with the
    'coordinate_names' parameter. The struct can have also other fields that can be read and written in the kernel, for
    example boundary parameters.

    :param assignments: list of update equations or AST nodes
    :param index_fields: list of index fields, i.e. 1D fields with struct data type
    :param type_info: see documentation of :func:`create_kernel`
    :param function_name: see documentation of :func:`create_kernel`
    :param coordinate_names: name of the coordinate fields in the struct data type
    :return: abstract syntax tree
    """
    raise NotImplementedError
