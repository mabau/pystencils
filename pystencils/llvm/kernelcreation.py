from pystencils.llvm.llvmjit import make_python_function
from pystencils.transformations import insert_casts


def create_kernel(assignments, function_name="kernel", type_info=None, split_groups=(),
                  iteration_slice=None, ghost_layers=None):
    """
    Creates an abstract syntax tree for a kernel function, by taking a list of update rules.

    Loops are created according to the field accesses in the equations.

    Args:
    assignments: list of sympy equations, containing accesses to :class:`pystencils.field.Field`.
                 Defining the update rules of the kernel
    function_name: name of the generated function - only important if generated code is written out
                   type_info: a map from symbol name to a C type specifier. If not specified all symbols are assumed to
                   be of type 'double' except symbols which occur on the left hand side of equations where the
                   right hand side is a sympy Boolean which are assumed to be 'bool' .
    split_groups: Specification on how to split up inner loop into multiple loops. For details see
                  transformation :func:`pystencils.transformation.split_inner_loop`
    iteration_slice: if not None, iteration is done only over this slice of the field
    ghost_layers: a sequence of pairs for each coordinate with lower and upper nr of ghost layers
                 if None, the number of ghost layers is determined automatically and assumed to be equal for a
                 all dimensions

    :return: :class:`pystencils.ast.KernelFunction` node
    """
    from pystencils.cpu import create_kernel
    code = create_kernel(assignments, function_name, type_info, split_groups, iteration_slice, ghost_layers)
    code.body = insert_casts(code.body)
    code._compile_function = make_python_function
    code._backend = 'llvm'
    return code
