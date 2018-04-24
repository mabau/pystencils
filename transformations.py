import warnings
from collections import defaultdict, OrderedDict, namedtuple
from copy import deepcopy
from types import MappingProxyType
import sympy as sp
from sympy.logic.boolalg import Boolean
from sympy.tensor import IndexedBase
from pystencils.assignment import Assignment
from pystencils.field import Field, FieldType, offset_component_to_direction_string
from pystencils.data_types import TypedSymbol, PointerType, StructType, get_base_type, cast_func, \
    pointer_arithmetic_func, get_type_of_expression, collate_types
from pystencils.slicing import normalize_slice
import pystencils.astnodes as ast


def filtered_tree_iteration(node, node_type):
    for arg in node.args:
        if isinstance(arg, node_type):
            yield arg
        yield from filtered_tree_iteration(arg, node_type)


def unify_shape_symbols(body, common_shape, fields):
    """Replaces symbols for array sizes to ensure they are represented by the same unique symbol.

    When creating a kernel with variable array sizes, all passed arrays must have the same size.
    This is ensured when the kernel is called. Inside the kernel this means that only on symbol has to be used instead
    of one for each field. For example shape_arr1[0]  and shape_arr2[0] must be equal, so they should also be
    represented by the same symbol.

    Args:
        body: ast node, for the kernel part where substitutions is made, is modified in-place
        common_shape: shape of the field that was chosen
        fields: all fields whose shapes should be replaced by common_shape
    """
    substitutions = {}
    for field in fields:
        assert len(field.spatial_shape) == len(common_shape)
        if not field.has_fixed_shape:
            for common_shape_component, shape_component in zip(common_shape, field.spatial_shape):
                if shape_component != common_shape_component:
                    substitutions[shape_component] = common_shape_component
    if substitutions:
        body.subs(substitutions)


def get_common_shape(field_set):
    """Takes a set of pystencils Fields and returns their common spatial shape if it exists. Otherwise
    ValueError is raised"""
    nr_of_fixed_shaped_fields = 0
    for f in field_set:
        if f.has_fixed_shape:
            nr_of_fixed_shaped_fields += 1

    if nr_of_fixed_shaped_fields > 0 and nr_of_fixed_shaped_fields != len(field_set):
        fixed_field_names = ",".join([f.name for f in field_set if f.has_fixed_shape])
        var_field_names = ",".join([f.name for f in field_set if not f.has_fixed_shape])
        msg = "Mixing fixed-shaped and variable-shape fields in a single kernel is not possible\n"
        msg += "Variable shaped: %s \nFixed shaped:    %s" % (var_field_names, fixed_field_names)
        raise ValueError(msg)

    shape_set = set([f.spatial_shape for f in field_set])
    if nr_of_fixed_shaped_fields == len(field_set):
        if len(shape_set) != 1:
            raise ValueError("Differently sized field accesses in loop body: " + str(shape_set))

    shape = list(sorted(shape_set, key=lambda e: str(e[0])))[0]
    return shape


def make_loop_over_domain(body, function_name, iteration_slice=None, ghost_layers=None, loop_order=None):
    """Uses :class:`pystencils.field.Field.Access` to create (multiple) loops around given AST.

    Args:
        body: Block object with inner loop contents
        function_name: name of generated C function
        iteration_slice: if not None, iteration is done only over this slice of the field
        ghost_layers: a sequence of pairs for each coordinate with lower and upper nr of ghost layers
             if None, the number of ghost layers is determined automatically and assumed to be equal for a
             all dimensions
        loop_order: loop ordering from outer to inner loop (optimal ordering is same as layout)

    Returns:
        :class:`LoopOverCoordinate` instance with nested loops, ordered according to field layouts
    """
    # find correct ordering by inspecting participating FieldAccesses
    field_accesses = body.atoms(Field.Access)
    # exclude accesses to buffers from field_list, because buffers are treated separately
    field_list = [e.field for e in field_accesses if not FieldType.is_buffer(e.field)]
    fields = set(field_list)
    num_buffer_accesses = len(field_accesses) - len(field_list)

    if loop_order is None:
        loop_order = get_optimal_loop_ordering(fields)

    shape = get_common_shape(fields)
    unify_shape_symbols(body, common_shape=shape, fields=fields)

    if iteration_slice is not None:
        iteration_slice = normalize_slice(iteration_slice, shape)

    if ghost_layers is None:
        required_ghost_layers = max([fa.required_ghost_layers for fa in field_accesses])
        ghost_layers = [(required_ghost_layers, required_ghost_layers)] * len(loop_order)
    if isinstance(ghost_layers, int):
        ghost_layers = [(ghost_layers, ghost_layers)] * len(loop_order)

    def get_loop_stride(loop_begin, loop_end, step):
        return (loop_end - loop_begin) / step

    loop_strides = []
    loop_vars = []
    current_body = body
    for i, loop_coordinate in enumerate(reversed(loop_order)):
        if iteration_slice is None:
            begin = ghost_layers[loop_coordinate][0]
            end = shape[loop_coordinate] - ghost_layers[loop_coordinate][1]
            new_loop = ast.LoopOverCoordinate(current_body, loop_coordinate, begin, end, 1)
            current_body = ast.Block([new_loop])
            loop_strides.append(get_loop_stride(begin, end, 1))
            loop_vars.append(new_loop.loop_counter_symbol)
        else:
            slice_component = iteration_slice[loop_coordinate]
            if type(slice_component) is slice:
                sc = slice_component
                new_loop = ast.LoopOverCoordinate(current_body, loop_coordinate, sc.start, sc.stop, sc.step)
                current_body = ast.Block([new_loop])
                loop_strides.append(get_loop_stride(sc.start, sc.stop, sc.step))
                loop_vars.append(new_loop.loop_counter_symbol)
            else:
                assignment = ast.SympyAssignment(ast.LoopOverCoordinate.get_loop_counter_symbol(loop_coordinate),
                                                 sp.sympify(slice_component))
                current_body.insert_front(assignment)

    loop_vars = [num_buffer_accesses * var for var in loop_vars]
    ast_node = ast.KernelFunction(current_body, ghost_layers=ghost_layers, function_name=function_name, backend='cpu')
    return ast_node, loop_strides, loop_vars


def create_intermediate_base_pointer(field_access, coordinates, previous_ptr):
    r"""
    Addressing elements in structured arrays is done with :math:`ptr\left[ \sum_i c_i \cdot s_i \right]`
    where :math:`c_i` is the coordinate value and :math:`s_i` the stride of a coordinate.
    The sum can be split up into multiple parts, such that parts of it can be pulled before loops.
    This function creates such an access for coordinates :math:`i \in \mbox{coordinates}`.
    Returns a new typed symbol, where the name encodes which coordinates have been resolved.

    Args:
        field_access: instance of :class:`pystencils.field.Field.Access` which provides strides and offsets
        coordinates: mapping of coordinate ids to its value, where stride*value is calculated
        previous_ptr: the pointer which is de-referenced

    Returns
        tuple with the new pointer symbol and the calculated offset

    Examples:
        >>> field = Field.create_generic('myfield', spatial_dimensions=2, index_dimensions=1)
        >>> x, y = sp.symbols("x y")
        >>> prev_pointer = TypedSymbol("ptr", "double")
        >>> create_intermediate_base_pointer(field[1,-2](5), {0: x}, prev_pointer)
        (ptr_E, x*fstride_myfield[0] + fstride_myfield[0])
        >>> create_intermediate_base_pointer(field[1,-2](5), {0: x, 1 : y }, prev_pointer)
        (ptr_E_2S, x*fstride_myfield[0] + y*fstride_myfield[1] + fstride_myfield[0] - 2*fstride_myfield[1])
    """
    field = field_access.field
    offset = 0
    name = ""
    list_to_hash = []
    for coordinate_id, coordinate_value in coordinates.items():
        offset += field.strides[coordinate_id] * coordinate_value

        if coordinate_id < field.spatial_dimensions:
            offset += field.strides[coordinate_id] * field_access.offsets[coordinate_id]
            if type(field_access.offsets[coordinate_id]) is int:
                offset_comp = offset_component_to_direction_string(coordinate_id, field_access.offsets[coordinate_id])
                name += "_"
                name += offset_comp if offset_comp else "C"
            else:
                list_to_hash.append(field_access.offsets[coordinate_id])
        else:
            if type(coordinate_value) is int:
                name += "_%d" % (coordinate_value,)
            else:
                list_to_hash.append(coordinate_value)

    if len(list_to_hash) > 0:
        name += "%0.6X" % (abs(hash(tuple(list_to_hash))))

    new_ptr = TypedSymbol(previous_ptr.name + name, previous_ptr.dtype)

    return new_ptr, offset


def parse_base_pointer_info(base_pointer_specification, loop_order, field):
    """
    Creates base pointer specification for :func:`resolve_field_accesses` function.

    Specification of how many and which intermediate pointers are created for a field access.
    For example [ (0), (2,3,)]  creates on base pointer for coordinates 2 and 3 and writes the offset for coordinate
    zero directly in the field access. These specifications are defined dependent on the loop ordering.
    This function translates more readable version into the specification above.

    Allowed specifications:
        - "spatialInner<int>" spatialInner0 is the innermost loop coordinate,
          spatialInner1 the loop enclosing the innermost
        - "spatialOuter<int>" spatialOuter0 is the outermost loop
        - "index<int>": index coordinate
        - "<int>": specifying directly the coordinate

    Args:
        base_pointer_specification: nested list with above specifications
        loop_order: list with ordering of loops from outer to inner
        field:

    Returns:
        list of tuples that can be passed to :func:`resolve_field_accesses`
    """
    result = []
    specified_coordinates = set()
    loop_order = list(reversed(loop_order))
    for spec_group in base_pointer_specification:
        new_group = []

        def add_new_element(elem):
            if elem >= field.spatial_dimensions + field.index_dimensions:
                raise ValueError("Coordinate %d does not exist" % (elem,))
            new_group.append(elem)
            if elem in specified_coordinates:
                raise ValueError("Coordinate %d specified two times" % (elem,))
            specified_coordinates.add(elem)
        for element in spec_group:
            if type(element) is int:
                add_new_element(element)
            elif element.startswith("spatial"):
                element = element[len("spatial"):]
                if element.startswith("Inner"):
                    index = int(element[len("Inner"):])
                    add_new_element(loop_order[index])
                elif element.startswith("Outer"):
                    index = int(element[len("Outer"):])
                    add_new_element(loop_order[-index])
                elif element == "all":
                    for i in range(field.spatial_dimensions):
                        add_new_element(i)
                else:
                    raise ValueError("Could not parse " + element)
            elif element.startswith("index"):
                index = int(element[len("index"):])
                add_new_element(field.spatial_dimensions + index)
            else:
                raise ValueError("Unknown specification %s" % (element,))

        result.append(new_group)

    all_coordinates = set(range(field.spatial_dimensions + field.index_dimensions))
    rest = all_coordinates - specified_coordinates
    if rest:
        result.append(list(rest))

    return result


def substitute_array_accesses_with_constants(ast_node):
    """Substitutes all instances of Indexed (array accesses) that are not field accesses with constants.
    Benchmarks showed that using an array access as loop bound or in pointer computations cause some compilers to do
    less optimizations.
    This transformation should be after field accesses have been resolved (since they introduce array accesses) and
    before constants are moved before the loops.
    """

    def handle_sympy_expression(expr, parent_block):
        """Returns sympy expression where array accesses have been replaced with constants, together with a list
        of assignments that define these constants"""
        if not isinstance(expr, sp.Expr):
            return expr

        # get all indexed expressions that are not field accesses
        indexed_expressions = [e for e in expr.atoms(sp.Indexed) if not isinstance(e, ast.ResolvedFieldAccess)]

        # special case: right hand side is a single indexed expression, then nothing has to be done
        if len(indexed_expressions) == 1 and expr == indexed_expressions[0]:
            return expr

        constants_definitions = []
        constant_substitutions = {}
        for indexed_expr in indexed_expressions:
            base, idx = indexed_expr.args
            typed_symbol = base.args[0]
            base_type = deepcopy(get_base_type(typed_symbol.dtype))
            base_type.const = False
            constant_replacing_indexed = TypedSymbol(typed_symbol.name + str(idx), base_type)
            constants_definitions.append(ast.SympyAssignment(constant_replacing_indexed, indexed_expr))
            constant_substitutions[indexed_expr] = constant_replacing_indexed
        constants_definitions.sort(key=lambda e: e.lhs.name)

        already_defined = parent_block.symbols_defined
        for new_assignment in constants_definitions:
            if new_assignment.lhs not in already_defined:
                parent_block.insert_before(new_assignment, ast_node)

        return expr.subs(constant_substitutions)

    if isinstance(ast_node, ast.SympyAssignment):
        ast_node.rhs = handle_sympy_expression(ast_node.rhs, ast_node.parent)
        ast_node.lhs = handle_sympy_expression(ast_node.lhs, ast_node.parent)
    elif isinstance(ast_node, ast.LoopOverCoordinate):
        ast_node.start = handle_sympy_expression(ast_node.start, ast_node.parent)
        ast_node.stop = handle_sympy_expression(ast_node.stop, ast_node.parent)
        ast_node.step = handle_sympy_expression(ast_node.step, ast_node.parent)
        substitute_array_accesses_with_constants(ast_node.body)
    else:
        for a in ast_node.args:
            substitute_array_accesses_with_constants(a)


def resolve_buffer_accesses(ast_node, base_buffer_index, read_only_field_names=set()):
    def visit_sympy_expr(expr, enclosing_block, sympy_assignment):
        if isinstance(expr, Field.Access):
            field_access = expr

            # Do not apply transformation if field is not a buffer
            if not FieldType.is_buffer(field_access.field):
                return expr

            buffer = field_access.field

            dtype = PointerType(buffer.dtype, const=buffer.name in read_only_field_names, restrict=True)
            field_ptr = TypedSymbol("%s%s" % (Field.DATA_PREFIX, symbol_name_to_variable_name(buffer.name)), dtype)

            buffer_index = base_buffer_index
            if len(field_access.index) > 1:
                raise RuntimeError('Only indexing dimensions up to 1 are currently supported in buffers!')

            if len(field_access.index) > 0:
                cell_index = field_access.index[0]
                buffer_index += cell_index

            result = ast.ResolvedFieldAccess(field_ptr, buffer_index, field_access.field, field_access.offsets,
                                             field_access.index)

            return visit_sympy_expr(result, enclosing_block, sympy_assignment)
        else:
            if isinstance(expr, ast.ResolvedFieldAccess):
                return expr

            new_args = [visit_sympy_expr(e, enclosing_block, sympy_assignment) for e in expr.args]
            kwargs = {'evaluate': False} if type(expr) in (sp.Add, sp.Mul, sp.Piecewise) else {}
            return expr.func(*new_args, **kwargs) if new_args else expr

    def visit_node(sub_ast):
        if isinstance(sub_ast, ast.SympyAssignment):
            enclosing_block = sub_ast.parent
            assert type(enclosing_block) is ast.Block
            sub_ast.lhs = visit_sympy_expr(sub_ast.lhs, enclosing_block, sub_ast)
            sub_ast.rhs = visit_sympy_expr(sub_ast.rhs, enclosing_block, sub_ast)
        else:
            for i, a in enumerate(sub_ast.args):
                visit_node(a)

    return visit_node(ast_node)


def resolve_field_accesses(ast_node, read_only_field_names=set(),
                           field_to_base_pointer_info=MappingProxyType({}),
                           field_to_fixed_coordinates=MappingProxyType({})):
    """
    Substitutes :class:`pystencils.field.Field.Access` nodes by array indexing

    Args:
        ast_node: the AST root
        read_only_field_names: set of field names which are considered read-only
        field_to_base_pointer_info: a list of tuples indicating which intermediate base pointers should be created
                                    for details see :func:`parse_base_pointer_info`
        field_to_fixed_coordinates: map of field name to a tuple of coordinate symbols. Instead of using the loop
                                    counters to index the field these symbols are used as coordinates

    Returns
        transformed AST
    """
    field_to_base_pointer_info = OrderedDict(sorted(field_to_base_pointer_info.items(), key=lambda pair: pair[0]))
    field_to_fixed_coordinates = OrderedDict(sorted(field_to_fixed_coordinates.items(), key=lambda pair: pair[0]))

    def visit_sympy_expr(expr, enclosing_block, sympy_assignment):
        if isinstance(expr, Field.Access):
            field_access = expr
            field = field_access.field

            if field.name in field_to_base_pointer_info:
                base_pointer_info = field_to_base_pointer_info[field.name]
            else:
                base_pointer_info = [list(range(field.index_dimensions + field.spatial_dimensions))]

            dtype = PointerType(field.dtype, const=field.name in read_only_field_names, restrict=True)
            field_ptr = TypedSymbol("%s%s" % (Field.DATA_PREFIX, symbol_name_to_variable_name(field.name)), dtype)

            def create_coordinate_dict(group_param):
                coordinates = {}
                for e in group_param:
                    if e < field.spatial_dimensions:
                        if field.name in field_to_fixed_coordinates:
                            coordinates[e] = field_to_fixed_coordinates[field.name][e]
                        else:
                            coordinates[e] = ast.LoopOverCoordinate.get_loop_counter_symbol(e)
                        coordinates[e] *= field.dtype.item_size
                    else:
                        if isinstance(field.dtype, StructType):
                            assert field.index_dimensions == 1
                            accessed_field_name = field_access.index[0]
                            assert isinstance(accessed_field_name, str)
                            coordinates[e] = field.dtype.get_element_offset(accessed_field_name)
                        else:
                            coordinates[e] = field_access.index[e - field.spatial_dimensions]

                return coordinates

            last_pointer = field_ptr

            for group in reversed(base_pointer_info[1:]):
                coord_dict = create_coordinate_dict(group)
                new_ptr, offset = create_intermediate_base_pointer(field_access, coord_dict, last_pointer)
                if new_ptr not in enclosing_block.symbols_defined:
                    new_assignment = ast.SympyAssignment(new_ptr, last_pointer + offset, is_const=False)
                    enclosing_block.insert_before(new_assignment, sympy_assignment)
                last_pointer = new_ptr

            coord_dict = create_coordinate_dict(base_pointer_info[0])
            _, offset = create_intermediate_base_pointer(field_access, coord_dict, last_pointer)
            result = ast.ResolvedFieldAccess(last_pointer, offset, field_access.field,
                                             field_access.offsets, field_access.index)

            if isinstance(get_base_type(field_access.field.dtype), StructType):
                new_type = field_access.field.dtype.get_element_type(field_access.index[0])
                result = cast_func(result, new_type)

            return visit_sympy_expr(result, enclosing_block, sympy_assignment)
        else:
            if isinstance(expr, ast.ResolvedFieldAccess):
                return expr

            new_args = [visit_sympy_expr(e, enclosing_block, sympy_assignment) for e in expr.args]
            kwargs = {'evaluate': False} if type(expr) in (sp.Add, sp.Mul, sp.Piecewise) else {}
            return expr.func(*new_args, **kwargs) if new_args else expr

    def visit_node(sub_ast):
        if isinstance(sub_ast, ast.SympyAssignment):
            enclosing_block = sub_ast.parent
            assert type(enclosing_block) is ast.Block
            sub_ast.lhs = visit_sympy_expr(sub_ast.lhs, enclosing_block, sub_ast)
            sub_ast.rhs = visit_sympy_expr(sub_ast.rhs, enclosing_block, sub_ast)
        else:
            for i, a in enumerate(sub_ast.args):
                visit_node(a)

    return visit_node(ast_node)


def move_constants_before_loop(ast_node):
    """
    Moves :class:`pystencils.ast.SympyAssignment` nodes out of loop body if they are iteration independent.
    Call this after creating the loop structure with :func:`make_loop_over_domain`
    :param ast_node:
    :return:
    """
    def find_block_to_move_to(node):
        """
        Traverses parents of node as long as the symbols are independent and returns a (parent) block
        the assignment can be safely moved to
        :param node: SympyAssignment inside a Block
        :return blockToInsertTo, childOfBlockToInsertBefore
        """
        assert isinstance(node, ast.SympyAssignment)
        assert isinstance(node.parent, ast.Block)

        last_block = node.parent
        last_block_child = node
        element = node.parent
        prev_element = node
        while element:
            if isinstance(element, ast.Block):
                last_block = element
                last_block_child = prev_element

            if isinstance(element, ast.Conditional):
                critical_symbols = element.condition_expr.atoms(sp.Symbol)
            else:
                critical_symbols = element.symbols_defined
            if node.undefined_symbols.intersection(critical_symbols):
                break
            prev_element = element
            element = element.parent
        return last_block, last_block_child

    def check_if_assignment_already_in_block(assignment, target_block):
        for arg in target_block.args:
            if type(arg) is not ast.SympyAssignment:
                continue
            if arg.lhs == assignment.lhs:
                return arg
        return None

    def get_blocks(node, result_list):
        if isinstance(node, ast.Block):
            result_list.insert(0, node)
        if isinstance(node, ast.Node):
            for a in node.args:
                get_blocks(a, result_list)

    all_blocks = []
    get_blocks(ast_node, all_blocks)
    for block in all_blocks:
        children = block.take_child_nodes()
        for child in children:
            if not isinstance(child, ast.SympyAssignment):
                block.append(child)
            else:
                target, child_to_insert_before = find_block_to_move_to(child)
                if target == block:     # movement not possible
                    target.append(child)
                else:
                    existing_assignment = check_if_assignment_already_in_block(child, target)
                    if not existing_assignment:
                        target.insert_before(child, child_to_insert_before)
                    else:
                        assert existing_assignment.rhs == child.rhs, "Symbol with same name exists already"


def split_inner_loop(ast_node: ast.Node, symbol_groups):
    """
    Splits inner loop into multiple loops to minimize the amount of simultaneous load/store streams

    Args:
        ast_node: AST root
        symbol_groups: sequence of symbol sequences: for each symbol sequence a new inner loop is created which
                       updates these symbols and their dependent symbols. Symbols which are in none of the symbolGroups
                       and which no symbol in a symbol group depends on, are not updated!
    """
    all_loops = ast_node.atoms(ast.LoopOverCoordinate)
    inner_loop = [l for l in all_loops if l.is_innermost_loop]
    assert len(inner_loop) == 1, "Error in AST: multiple innermost loops. Was split transformation already called?"
    inner_loop = inner_loop[0]
    assert type(inner_loop.body) is ast.Block
    outer_loop = [l for l in all_loops if l.is_outermost_loop]
    assert len(outer_loop) == 1, "Error in AST, multiple outermost loops."
    outer_loop = outer_loop[0]

    symbols_with_temporary_array = OrderedDict()
    assignment_map = OrderedDict((a.lhs, a) for a in inner_loop.body.args)

    assignment_groups = []
    for symbol_group in symbol_groups:
        # get all dependent symbols
        symbols_to_process = list(symbol_group)
        symbols_resolved = set()
        while symbols_to_process:
            s = symbols_to_process.pop()
            if s in symbols_resolved:
                continue

            if s in assignment_map:  # if there is no assignment inside the loop body it is independent already
                for new_symbol in assignment_map[s].rhs.atoms(sp.Symbol):
                    if type(new_symbol) is not Field.Access and new_symbol not in symbols_with_temporary_array:
                        symbols_to_process.append(new_symbol)
            symbols_resolved.add(s)

        for symbol in symbol_group:
            if type(symbol) is not Field.Access:
                assert type(symbol) is TypedSymbol
                new_ts = TypedSymbol(symbol.name, PointerType(symbol.dtype))
                symbols_with_temporary_array[symbol] = IndexedBase(new_ts, shape=(1,))[inner_loop.loop_counter_symbol]

        assignment_group = []
        for assignment in inner_loop.body.args:
            if assignment.lhs in symbols_resolved:
                new_rhs = assignment.rhs.subs(symbols_with_temporary_array.items())
                if type(assignment.lhs) is not Field.Access and assignment.lhs in symbol_group:
                    assert type(assignment.lhs) is TypedSymbol
                    new_ts = TypedSymbol(assignment.lhs.name, PointerType(assignment.lhs.dtype))
                    new_lhs = IndexedBase(new_ts, shape=(1,))[inner_loop.loop_counter_symbol]
                else:
                    new_lhs = assignment.lhs
                assignment_group.append(ast.SympyAssignment(new_lhs, new_rhs))
        assignment_groups.append(assignment_group)

    new_loops = [inner_loop.new_loop_with_different_body(ast.Block(group)) for group in assignment_groups]
    inner_loop.parent.replace(inner_loop, ast.Block(new_loops))

    for tmp_array in symbols_with_temporary_array:
        tmp_array_pointer = TypedSymbol(tmp_array.name, PointerType(tmp_array.dtype))
        outer_loop.parent.insert_front(ast.TemporaryMemoryAllocation(tmp_array_pointer, inner_loop.stop))
        outer_loop.parent.append(ast.TemporaryMemoryFree(tmp_array_pointer))


def cut_loop(loop_node, cutting_points):
    """Cuts loop at given cutting points, that means one loop is transformed into len(cuttingPoints)+1 new loops
    that range from  old_begin to cutting_points[1], ..., cutting_points[-1] to old_end"""
    if loop_node.step != 1:
        raise NotImplementedError("Can only split loops that have a step of 1")
    new_loops = []
    new_start = loop_node.start
    cutting_points = list(cutting_points) + [loop_node.stop]
    for new_end in cutting_points:
        if new_end - new_start == 1:
            new_body = deepcopy(loop_node.body)
            new_body.subs({loop_node.loop_counter_symbol: new_start})
            new_loops.append(new_body)
        else:
            new_loop = ast.LoopOverCoordinate(deepcopy(loop_node.body), loop_node.coordinate_to_loop_over,
                                              new_start, new_end, loop_node.step)
            new_loops.append(new_loop)
        new_start = new_end
    loop_node.parent.replace(loop_node, new_loops)


def simplify_conditionals(node: ast.Node, loop_counter_simplification: bool=False) -> None:
    """Removes conditionals that are always true/false.

    Args:
        node: ast node, all descendants of this node are simplified
        loop_counter_simplification: if enabled, tries to detect if a conditional is always true/false
                                     depending on the surrounding loop. For example if the surrounding loop goes from
                                     x=0 to 10 and the condition is x < 0, it is removed.
                                     This analysis needs the integer set library (ISL) islpy, so it is not done by
                                     default.
    """
    for conditional in node.atoms(ast.Conditional):
        conditional.condition_expr = sp.simplify(conditional.condition_expr)
        if conditional.condition_expr == sp.true:
            conditional.parent.replace(conditional, [conditional.true_block])
        elif conditional.condition_expr == sp.false:
            conditional.parent.replace(conditional, [conditional.false_block] if conditional.false_block else [])
        elif loop_counter_simplification:
            try:
                # noinspection PyUnresolvedReferences
                from pystencils.integer_set_analysis import simplify_loop_counter_dependent_conditional
                simplify_loop_counter_dependent_conditional(conditional)
            except ImportError:
                warnings.warn("Integer simplifications in conditionals skipped, because ISLpy package not installed")


def cleanup_blocks(node: ast.Node) -> None:
    """Curly Brace Removal: Removes empty blocks, and replaces blocks with a single child by its child """
    if isinstance(node, ast.SympyAssignment):
        return
    elif isinstance(node, ast.Block):
        for a in list(node.args):
            cleanup_blocks(a)
        if len(node.args) <= 1 and isinstance(node.parent, ast.Block):
            node.parent.replace(node, node.args)
            return
    else:
        for a in node.args:
            cleanup_blocks(a)


def symbol_name_to_variable_name(symbol_name):
    """Replaces characters which are allowed in sympy symbol names but not in C/C++ variable names"""
    return symbol_name.replace("^", "_")


class KernelConstraintsCheck:
    """Checks if the input to create_kernel is valid.

    Test the following conditions:

    - SSA Form for pure symbols:
        -  Every pure symbol may occur only once as left-hand-side of an assignment
        -  Every pure symbol that is read, may not be written to later
    - Independence / Parallelization condition:
        - a field that is written may only be read at exact the same spatial position

    (Pure symbols are symbols that are not Field.Accesses)
    """
    FieldAndIndex = namedtuple('FieldAndIndex', ['field', 'index'])

    def __init__(self, type_for_symbol, check_independence_condition):
        self._type_for_symbol = type_for_symbol
        self._defined_pure_symbols = set()
        self._accessed_pure_symbols = set()

        self._field_writes = defaultdict(set)
        self.fields_read = set()
        self.check_independence_condition = check_independence_condition

    def process_assignment(self, assignment):
        # for checks it is crucial to process rhs before lhs to catch e.g. a = a + 1
        new_rhs = self.process_expression(assignment.rhs)
        new_lhs = self._process_lhs(assignment.lhs)
        return ast.SympyAssignment(new_lhs, new_rhs)

    def process_expression(self, rhs):
        self._update_accesses_rhs(rhs)
        if isinstance(rhs, Field.Access):
            return rhs
        elif isinstance(rhs, TypedSymbol):
            return rhs
        elif isinstance(rhs, sp.Symbol):
            return TypedSymbol(symbol_name_to_variable_name(rhs.name), self._type_for_symbol[rhs.name])
        else:
            new_args = [self.process_expression(arg) for arg in rhs.args]
            return rhs.func(*new_args) if new_args else rhs

    @property
    def fields_written(self):
        return set(k.field for k, v in self._field_writes.items() if len(v))

    def _process_lhs(self, lhs):
        assert isinstance(lhs, sp.Symbol)
        self._update_accesses_lhs(lhs)
        if not isinstance(lhs, Field.Access) and not isinstance(lhs, TypedSymbol):
            return TypedSymbol(lhs.name, self._type_for_symbol[lhs.name])
        else:
            return lhs

    def _update_accesses_lhs(self, lhs):
        if isinstance(lhs, Field.Access):
            fai = self.FieldAndIndex(lhs.field, lhs.index)
            self._field_writes[fai].add(lhs.offsets)
            if len(self._field_writes[fai]) > 1:
                raise ValueError(f"Field {lhs.field.name} is written at two different locations")
        elif isinstance(lhs, sp.Symbol):
            if lhs in self._defined_pure_symbols:
                raise ValueError(f"Assignments not in SSA form, multiple assignments to {lhs.name}")
            if lhs in self._accessed_pure_symbols:
                raise ValueError(f"Symbol {lhs.name} is written, after it has been read")
            self._defined_pure_symbols.add(lhs)

    def _update_accesses_rhs(self, rhs):
        if isinstance(rhs, Field.Access) and self.check_independence_condition:
            writes = self._field_writes[self.FieldAndIndex(rhs.field, rhs.index)]
            for write_offset in writes:
                assert len(writes) == 1
                if write_offset != rhs.offsets:
                    raise ValueError(f"Violation of loop independence condition. "
                                     f"Field {rhs.field} is read at {rhs.offsets} and written at {write_offset}")
            self.fields_read.add(rhs.field)
        elif isinstance(rhs, sp.Symbol):
            self._accessed_pure_symbols.add(rhs)


def add_types(eqs, type_for_symbol, check_independence_condition):
    """Traverses AST and replaces every :class:`sympy.Symbol` by a :class:`pystencils.typedsymbol.TypedSymbol`.

    Additionally returns sets of all fields which are read/written

    Args:
        eqs: list of equations
        type_for_symbol: dict mapping symbol names to types. Types are strings of C types like 'int' or 'double'
        check_independence_condition: check that loop iterations are independent - this has to be skipped for indexed
                                      kernels

    Returns:
        ``fields_read, fields_written, typed_equations`` set of read fields, set of written fields,
         list of equations where symbols have been replaced by typed symbols
    """
    if isinstance(type_for_symbol, str) or not hasattr(type_for_symbol, '__getitem__'):
        type_for_symbol = typing_from_sympy_inspection(eqs, type_for_symbol)

    check = KernelConstraintsCheck(type_for_symbol, check_independence_condition)

    def visit(obj):
        if isinstance(obj, list) or isinstance(obj, tuple):
            return [visit(e) for e in obj]
        if isinstance(obj, sp.Eq) or isinstance(obj, ast.SympyAssignment) or isinstance(obj, Assignment):
            return check.process_assignment(obj)
        elif isinstance(obj, ast.Conditional):
            false_block = None if obj.false_block is None else visit(obj.false_block)
            return ast.Conditional(check.process_expression(obj.condition_expr),
                                   true_block=visit(obj.true_block), false_block=false_block)
        elif isinstance(obj, ast.Block):
            return ast.Block([visit(e) for e in obj.args])
        elif isinstance(obj, ast.Node) and not isinstance(obj, ast.LoopOverCoordinate):
            return obj
        else:
            raise ValueError("Invalid object in kernel " + str(type(obj)))

    typed_equations = visit(eqs)

    return check.fields_read, check.fields_written, typed_equations


def insert_casts(node):
    """Checks the types and inserts casts and pointer arithmetic where necessary

    :param node: the head node of the ast
    :return: modified ast
    """
    def cast(zipped_args_types, target_dtype):
        """
        Adds casts to the arguments if their type differs from the target type
        :param zipped_args_types: a zipped list of args and types
        :param target_dtype: The target data type
        :return: args with possible casts
        """
        casted_args = []
        for argument, data_type in zipped_args_types:
            if data_type.numpy_dtype != target_dtype.numpy_dtype:  # ignoring const
                casted_args.append(cast_func(argument, target_dtype))
            else:
                casted_args.append(argument)
        return casted_args

    def pointer_arithmetic(expr_args):
        """
        Creates a valid pointer arithmetic function
        :param expr_args: Arguments of the add expression
        :return: pointer_arithmetic_func
        """
        pointer = None
        new_args = []
        for arg, data_type in expr_args:
            if data_type.func is PointerType:
                assert pointer is None
                pointer = arg
        for arg, data_type in expr_args:
            if arg != pointer:
                assert data_type.is_int() or data_type.is_uint()
                new_args.append(arg)
        new_args = sp.Add(*new_args) if len(new_args) > 0 else new_args
        return pointer_arithmetic_func(pointer, new_args)

    if isinstance(node, sp.AtomicExpr):
        return node
    args = []
    for arg in node.args:
        args.append(insert_casts(arg))
    # TODO indexed, LoopOverCoordinate
    if node.func in (sp.Add, sp.Mul, sp.Or, sp.And, sp.Pow, sp.Eq, sp.Ne, sp.Lt, sp.Le, sp.Gt, sp.Ge):
        # TODO optimize pow, don't cast integer on double
        types = [get_type_of_expression(arg) for arg in args]
        assert len(types) > 0
        target = collate_types(types)
        zipped = list(zip(args, types))
        if target.func is PointerType:
            assert node.func is sp.Add
            return pointer_arithmetic(zipped)
        else:
            return node.func(*cast(zipped, target))
    elif node.func is ast.SympyAssignment:
        lhs = args[0]
        rhs = args[1]
        target = get_type_of_expression(lhs)
        if target.func is PointerType:
            return node.func(*args)  # TODO fix, not complete
        else:
            return node.func(lhs, *cast([(rhs, get_type_of_expression(rhs))], target))
    elif node.func is ast.ResolvedFieldAccess:
        return node
    elif node.func is ast.Block:
        for old_arg, new_arg in zip(node.args, args):
            node.replace(old_arg, new_arg)
        return node
    elif node.func is ast.LoopOverCoordinate:
        for old_arg, new_arg in zip(node.args, args):
            node.replace(old_arg, new_arg)
        return node
    elif node.func is sp.Piecewise:
        expressions = [expr for (expr, _) in args]
        types = [get_type_of_expression(expr) for expr in expressions]
        target = collate_types(types)
        zipped = list(zip(expressions, types))
        casted_expressions = cast(zipped, target)
        args = [arg.func(*[expr, arg.cond]) for (arg, expr) in zip(args, casted_expressions)]

    return node.func(*args)


def remove_conditionals_in_staggered_kernel(function_node: ast.KernelFunction) -> None:
    """Removes conditionals of a kernel that iterates over staggered positions by splitting the loops at last element"""

    all_inner_loops = [l for l in function_node.atoms(ast.LoopOverCoordinate) if l.is_innermost_loop]
    assert len(all_inner_loops) == 1, "Transformation works only on kernels with exactly one inner loop"
    inner_loop = all_inner_loops.pop()

    for loop in parents_of_type(inner_loop, ast.LoopOverCoordinate, include_current=True):
        cut_loop(loop, [loop.stop - 1])

    simplify_conditionals(function_node.body, loop_counter_simplification=True)
    cleanup_blocks(function_node.body)
    move_constants_before_loop(function_node.body)
    cleanup_blocks(function_node.body)


# --------------------------------------- Helper Functions -------------------------------------------------------------


def typing_from_sympy_inspection(eqs, default_type="double"):
    """
    Creates a default symbol name to type mapping.
    If a sympy Boolean is assigned to a symbol it is assumed to be 'bool' otherwise the default type, usually ('double')

    Args:
        eqs: list of equations
        default_type: the type for non-boolean symbols
    Returns:
        dictionary, mapping symbol name to type
    """
    result = defaultdict(lambda: default_type)
    for eq in eqs:
        if isinstance(eq, ast.Node):
            continue
        # problematic case here is when rhs is a symbol: then it is impossible to decide here without
        # further information what type the left hand side is - default fallback is the dict value then
        if isinstance(eq.rhs, Boolean) and not isinstance(eq.rhs, sp.Symbol):
            result[eq.lhs.name] = "bool"
    return result


def get_next_parent_of_type(node, parent_type):
    """Returns the next parent node of given type or None, if root is reached.

    Traverses the AST nodes parents until a parent of given type was found.
    If no such parent is found, None is returned
    """
    parent = node.parent
    while parent is not None:
        if isinstance(parent, parent_type):
            return parent
        parent = parent.parent
    return None


def parents_of_type(node, parent_type, include_current=False):
    """Generator for all parent nodes of given type"""
    parent = node if include_current else node.parent
    while parent is not None:
        if isinstance(parent, parent_type):
            yield parent
        parent = parent.parent


def get_optimal_loop_ordering(fields):
    """
    Determines the optimal loop order for a given set of fields.
    If the fields have different memory layout or different sizes an exception is thrown.

    Args:
        fields: sequence of fields

    Returns:
        list of coordinate ids, where the first list entry should be the outermost loop
    """
    assert len(fields) > 0
    ref_field = next(iter(fields))
    for field in fields:
        if field.spatial_dimensions != ref_field.spatial_dimensions:
            raise ValueError("All fields have to have the same number of spatial dimensions. Spatial field dimensions: "
                             + str({f.name: f.spatial_shape for f in fields}))

    layouts = set([field.layout for field in fields])
    if len(layouts) > 1:
        raise ValueError("Due to different layout of the fields no optimal loop ordering exists " +
                         str({f.name: f.layout for f in fields}))
    layout = list(layouts)[0]
    return list(layout)


def get_loop_hierarchy(ast_node):
    """Determines the loop structure around a given AST node, i.e. the node has to be inside the loops.

    Returns:
        sequence of LoopOverCoordinate nodes, starting from outer loop to innermost loop
    """
    result = []
    node = ast_node
    while node is not None:
        node = get_next_parent_of_type(node, ast.LoopOverCoordinate)
        if node:
            result.append(node.coordinate_to_loop_over)
    return reversed(result)
