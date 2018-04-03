from pystencils.astnodes import SympyAssignment, Block, LoopOverCoordinate, KernelFunction
from pystencils.transformations import resolve_field_accesses, \
    type_all_equations, move_constants_before_loop, insert_casts
from pystencils.data_types import TypedSymbol, BasicType, StructType
from pystencils.field import FieldType
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
    fields_read, fields_written, assignments = type_all_equations(assignments, type_info)
    all_fields = fields_read.union(fields_written)

    for indexField in index_fields:
        indexField.fieldType = FieldType.INDEXED
        assert FieldType.is_indexed(indexField)
        assert indexField.spatial_dimensions == 1, "Index fields have to be 1D"

    non_index_fields = [f for f in all_fields if f not in index_fields]
    spatial_coordinates = {f.spatial_dimensions for f in non_index_fields}
    assert len(spatial_coordinates) == 1, "Non-index fields do not have the same number of spatial coordinates"
    spatial_coordinates = list(spatial_coordinates)[0]

    def get_coordinate_symbol_assignment(name):
        for index_field in index_fields:
            assert isinstance(index_field.dtype, StructType), "Index fields have to have a struct datatype"
            data_type = index_field.dtype
            if data_type.has_element(name):
                rhs = index_field[0](name)
                lhs = TypedSymbol(name, BasicType(data_type.get_element_type(name)))
                return SympyAssignment(lhs, rhs)
        raise ValueError("Index %s not found in any of the passed index fields" % (name,))

    coordinate_symbol_assignments = [get_coordinate_symbol_assignment(n)
                                     for n in coordinate_names[:spatial_coordinates]]
    coordinate_typed_symbols = [eq.lhs for eq in coordinate_symbol_assignments]
    assignments = coordinate_symbol_assignments + assignments

    # make 1D loop over index fields
    loop_body = Block([])
    loop_node = LoopOverCoordinate(loop_body, coordinate_to_loop_over=0, start=0, stop=index_fields[0].shape[0])

    for assignment in assignments:
        loop_body.append(assignment)

    function_body = Block([loop_node])
    ast = KernelFunction(function_body, None, function_name, backend='llvm')

    read_only_fields = set([f.name for f in fields_read - fields_written])
    fixed_coordinate_mapping = {f.name: coordinate_typed_symbols for f in non_index_fields}
    resolve_field_accesses(ast, read_only_fields, field_to_fixed_coordinates=fixed_coordinate_mapping)
    move_constants_before_loop(ast)

    desympy_ast(ast)
    insert_casts(ast)
    ast.compile = partial(make_python_function, ast)

    return ast
