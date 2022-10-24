import hashlib
import pickle
import warnings
from collections import OrderedDict
from copy import deepcopy
from types import MappingProxyType

import sympy as sp

import pystencils.astnodes as ast
from pystencils.assignment import Assignment
from pystencils.typing import (CastFunc, PointerType, StructType, TypedSymbol, get_base_type,
                               ReinterpretCastFunc, get_next_parent_of_type, parents_of_type)
from pystencils.field import Field, FieldType
from pystencils.typing import FieldPointerSymbol
from pystencils.sympyextensions import fast_subs
from pystencils.simp.assignment_collection import AssignmentCollection
from pystencils.slicing import normalize_slice
from pystencils.integer_functions import int_div


class NestedScopes:
    """Symbol visibility model using nested scopes

    - every accessed symbol that was not defined before, is added as a "free parameter"
    - free parameters are global, i.e. they are not in scopes
    - push/pop adds or removes a scope

    >>> s = NestedScopes()
    >>> s.access_symbol("a")
    >>> s.is_defined("a")
    False
    >>> s.free_parameters
    {'a'}
    >>> s.define_symbol("b")
    >>> s.is_defined("b")
    True
    >>> s.push()
    >>> s.is_defined_locally("b")
    False
    >>> s.define_symbol("c")
    >>> s.pop()
    >>> s.is_defined("c")
    False
    """

    def __init__(self):
        self.free_parameters = set()
        self._defined = [set()]

    def access_symbol(self, symbol):
        if not self.is_defined(symbol):
            self.free_parameters.add(symbol)

    def define_symbol(self, symbol):
        self._defined[-1].add(symbol)

    def is_defined(self, symbol):
        return any(symbol in scopes for scopes in self._defined)

    def is_defined_locally(self, symbol):
        return symbol in self._defined[-1]

    def push(self):
        self._defined.append(set())

    def pop(self):
        self._defined.pop()
        assert self.depth >= 1

    @property
    def depth(self):
        return len(self._defined)


def filtered_tree_iteration(node, node_type, stop_type=None):
    for arg in node.args:
        if isinstance(arg, node_type):
            yield arg
        elif stop_type and isinstance(node, stop_type):
            continue

        yield from filtered_tree_iteration(arg, node_type)


def generic_visit(term, visitor):
    if isinstance(term, AssignmentCollection):
        new_main_assignments = generic_visit(term.main_assignments, visitor)
        new_subexpressions = generic_visit(term.subexpressions, visitor)
        return term.copy(new_main_assignments, new_subexpressions)
    elif isinstance(term, list):
        return [generic_visit(e, visitor) for e in term]
    elif isinstance(term, Assignment):
        return Assignment(term.lhs, generic_visit(term.rhs, visitor))
    elif isinstance(term, sp.Matrix):
        return term.applyfunc(lambda e: generic_visit(e, visitor))
    else:
        return visitor(term)


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
        msg += f"Variable shaped: {var_field_names} \nFixed shaped:    {fixed_field_names}"
        raise ValueError(msg)

    shape_set = set([f.spatial_shape for f in field_set])
    if nr_of_fixed_shaped_fields == len(field_set):
        if len(shape_set) != 1:
            raise ValueError("Differently sized field accesses in loop body: " + str(shape_set))

    shape = list(sorted(shape_set, key=lambda e: str(e[0])))[0]
    return shape


def make_loop_over_domain(body, iteration_slice=None, ghost_layers=None, loop_order=None):
    """Uses :class:`pystencils.field.Field.Access` to create (multiple) loops around given AST.

    Args:
        body: Block object with inner loop contents
        iteration_slice: if not None, iteration is done only over this slice of the field
        ghost_layers: a sequence of pairs for each coordinate with lower and upper nr of ghost layers
             if None, the number of ghost layers is determined automatically and assumed to be equal for a
             all dimensions
        loop_order: loop ordering from outer to inner loop (optimal ordering is same as layout)

    Returns:
        tuple of loop-node, ghost_layer_info
    """
    # find correct ordering by inspecting participating FieldAccesses
    field_accesses = body.atoms(Field.Access)
    field_accesses = {e for e in field_accesses if not e.is_absolute_access}

    # exclude accesses to buffers from field_list, because buffers are treated separately
    field_list = [e.field for e in field_accesses if not (FieldType.is_buffer(e.field) or FieldType.is_custom(e.field))]
    if len(field_list) == 0:  # when kernel contains only custom fields
        field_list = [e.field for e in field_accesses if not (FieldType.is_buffer(e.field))]

    fields = set(field_list)

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

    current_body = body
    for i, loop_coordinate in enumerate(reversed(loop_order)):
        if iteration_slice is None:
            begin = ghost_layers[loop_coordinate][0]
            end = shape[loop_coordinate] - ghost_layers[loop_coordinate][1]
            new_loop = ast.LoopOverCoordinate(current_body, loop_coordinate, begin, end, 1)
            current_body = ast.Block([new_loop])
        else:
            slice_component = iteration_slice[loop_coordinate]
            if type(slice_component) is slice:
                sc = slice_component
                new_loop = ast.LoopOverCoordinate(current_body, loop_coordinate, sc.start, sc.stop, sc.step)
                current_body = ast.Block([new_loop])
            else:
                assignment = ast.SympyAssignment(ast.LoopOverCoordinate.get_loop_counter_symbol(loop_coordinate),
                                                 sp.sympify(slice_component))
                current_body.insert_front(assignment)

    return current_body, ghost_layers


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
        (ptr_01, _stride_myfield_0*x + _stride_myfield_0)
        >>> create_intermediate_base_pointer(field[1,-2](5), {0: x, 1 : y }, prev_pointer)
        (ptr_01_1m2, _stride_myfield_0*x + _stride_myfield_0 + _stride_myfield_1*y - 2*_stride_myfield_1)
    """
    field = field_access.field
    offset = 0
    name = ""
    list_to_hash = []
    for coordinate_id, coordinate_value in coordinates.items():
        offset += field.strides[coordinate_id] * coordinate_value

        if coordinate_id < field.spatial_dimensions:
            offset += field.strides[coordinate_id] * field_access.offsets[coordinate_id]
            if field_access.offsets[coordinate_id].is_Integer:
                name += "_%d%d" % (coordinate_id, field_access.offsets[coordinate_id])
            else:
                list_to_hash.append(field_access.offsets[coordinate_id])
        else:
            if type(coordinate_value) is int:
                name += "_%d%d" % (coordinate_id, coordinate_value)
            else:
                list_to_hash.append(coordinate_value)

    if len(list_to_hash) > 0:
        name += hashlib.md5(pickle.dumps(list_to_hash)).hexdigest()[:16]

    name = name.replace("-", 'm')
    new_ptr = TypedSymbol(previous_ptr.name + name, previous_ptr.dtype)
    return new_ptr, offset


def parse_base_pointer_info(base_pointer_specification, loop_order, spatial_dimensions, index_dimensions):
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
        spatial_dimensions: number of spatial dimensions
        index_dimensions: number of index dimensions

    Returns:
        list of tuples that can be passed to :func:`resolve_field_accesses`

    Examples:
        >>> parse_base_pointer_info([['spatialOuter0'], ['index0']], loop_order=[2,1,0],
        ...                         spatial_dimensions=3, index_dimensions=1)
        [[0], [3], [1, 2]]
    """
    result = []
    specified_coordinates = set()
    loop_order = list(reversed(loop_order))
    for spec_group in base_pointer_specification:
        new_group = []

        def add_new_element(elem):
            if elem >= spatial_dimensions + index_dimensions:
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
                    for i in range(spatial_dimensions):
                        add_new_element(i)
                else:
                    raise ValueError("Could not parse " + element)
            elif element.startswith("index"):
                index = int(element[len("index"):])
                add_new_element(spatial_dimensions + index)
            else:
                raise ValueError(f"Unknown specification {element}")

        result.append(new_group)

    all_coordinates = set(range(spatial_dimensions + index_dimensions))
    rest = all_coordinates - specified_coordinates
    if rest:
        result.append(list(rest))

    return result


def get_base_buffer_index(ast_node, loop_counters=None, loop_iterations=None):
    """Used for buffer fields to determine the linearized index of the buffer dependent on loop counter symbols.

    Args:
        ast_node: ast before any field accesses are resolved
        loop_counters: for CPU kernels: leave to default 'None' (can be determined from loop nodes)
                       for GPU kernels: list of 'loop counters' from inner to outer loop
        loop_iterations: number of iterations of each loop from inner to outer, for CPU kernels leave to default

    Returns:
        base buffer index - required by 'resolve_buffer_accesses' function
    """
    if loop_counters is None or loop_iterations is None:
        loops = [l for l in filtered_tree_iteration(ast_node, ast.LoopOverCoordinate, ast.SympyAssignment)]
        loops.reverse()
        parents_of_innermost_loop = list(parents_of_type(loops[0], ast.LoopOverCoordinate, include_current=True))
        assert len(loops) == len(parents_of_innermost_loop)
        assert all(l1 is l2 for l1, l2 in zip(loops, parents_of_innermost_loop))

        actual_sizes = [int_div((loop.stop - loop.start), loop.step)
                        if loop.step != 1 else loop.stop - loop.start for loop in loops]

        actual_steps = [int_div((loop.loop_counter_symbol - loop.start), loop.step)
                        if loop.step != 1 else loop.loop_counter_symbol - loop.start for loop in loops]

    else:
        actual_sizes = loop_iterations
        actual_steps = loop_counters

    field_accesses = ast_node.atoms(Field.Access)
    buffer_accesses = {fa for fa in field_accesses if FieldType.is_buffer(fa.field)}
    buffer_index_size = len(buffer_accesses)

    base_buffer_index = actual_steps[0]
    actual_stride = 1
    for idx, actual_step in enumerate(actual_steps[1:]):
        cur_stride = actual_sizes[idx]
        actual_stride *= int(cur_stride) if isinstance(cur_stride, float) else cur_stride
        base_buffer_index += actual_stride * actual_step
    return base_buffer_index * buffer_index_size


def resolve_buffer_accesses(ast_node, base_buffer_index, read_only_field_names=None):

    if read_only_field_names is None:
        read_only_field_names = set()

    def visit_sympy_expr(expr, enclosing_block, sympy_assignment):
        if isinstance(expr, Field.Access):
            field_access = expr

            # Do not apply transformation if field is not a buffer
            if not FieldType.is_buffer(field_access.field):
                return expr

            buffer = field_access.field
            field_ptr = FieldPointerSymbol(buffer.name, buffer.dtype, const=buffer.name in read_only_field_names)

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


def resolve_field_accesses(ast_node, read_only_field_names=None,
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
    if read_only_field_names is None:
        read_only_field_names = set()
    field_to_base_pointer_info = OrderedDict(sorted(field_to_base_pointer_info.items(), key=lambda pair: pair[0]))
    field_to_fixed_coordinates = OrderedDict(sorted(field_to_fixed_coordinates.items(), key=lambda pair: pair[0]))

    def visit_sympy_expr(expr, enclosing_block, sympy_assignment):
        if isinstance(expr, Field.Access):
            field_access = expr
            field = field_access.field

            if field_access.indirect_addressing_fields:
                new_offsets = tuple(visit_sympy_expr(off, enclosing_block, sympy_assignment)
                                    for off in field_access.offsets)
                new_indices = tuple(visit_sympy_expr(ind, enclosing_block, sympy_assignment)
                                    if isinstance(ind, sp.Basic) else ind
                                    for ind in field_access.index)
                field_access = Field.Access(field_access.field, new_offsets,
                                            new_indices, field_access.is_absolute_access)

            if field.name in field_to_base_pointer_info:
                base_pointer_info = field_to_base_pointer_info[field.name]
            else:
                base_pointer_info = [list(range(field.index_dimensions + field.spatial_dimensions))]

            field_ptr = FieldPointerSymbol(
                field.name,
                field.dtype,
                const=field.name in read_only_field_names)

            def create_coordinate_dict(group_param):
                coordinates = {}
                for e in group_param:
                    if e < field.spatial_dimensions:
                        if field.name in field_to_fixed_coordinates:
                            if not field_access.is_absolute_access:
                                coordinates[e] = field_to_fixed_coordinates[field.name][e]
                            else:
                                coordinates[e] = 0
                        else:
                            if not field_access.is_absolute_access:
                                coordinates[e] = ast.LoopOverCoordinate.get_loop_counter_symbol(e)
                            else:
                                coordinates[e] = 0
                        coordinates[e] *= field.dtype.item_size
                    else:
                        if isinstance(field.dtype, StructType):
                            assert field.index_dimensions == 1
                            accessed_field_name = field_access.index[0]
                            if isinstance(accessed_field_name, sp.Symbol):
                                accessed_field_name = accessed_field_name.name
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
                accessed_field_name = field_access.index[0]
                if isinstance(accessed_field_name, sp.Symbol):
                    accessed_field_name = accessed_field_name.name
                new_type = field_access.field.dtype.get_element_type(accessed_field_name)
                result = ReinterpretCastFunc(result, new_type)

            return visit_sympy_expr(result, enclosing_block, sympy_assignment)
        else:
            if isinstance(expr, ast.ResolvedFieldAccess):
                return expr

            if hasattr(expr, 'args'):
                new_args = [visit_sympy_expr(e, enclosing_block, sympy_assignment) for e in expr.args]
            else:
                new_args = []
            kwargs = {'evaluate': False} if type(expr) in (sp.Add, sp.Mul, sp.Piecewise) else {}
            return expr.func(*new_args, **kwargs) if new_args else expr

    def visit_node(sub_ast):
        if isinstance(sub_ast, ast.SympyAssignment):
            enclosing_block = sub_ast.parent
            assert type(enclosing_block) is ast.Block
            sub_ast.lhs = visit_sympy_expr(sub_ast.lhs, enclosing_block, sub_ast)
            sub_ast.rhs = visit_sympy_expr(sub_ast.rhs, enclosing_block, sub_ast)
        elif isinstance(sub_ast, ast.Conditional):
            enclosing_block = sub_ast.parent
            assert type(enclosing_block) is ast.Block
            sub_ast.condition_expr = visit_sympy_expr(sub_ast.condition_expr, enclosing_block, sub_ast)
            visit_node(sub_ast.true_block)
            if sub_ast.false_block:
                visit_node(sub_ast.false_block)
        else:
            if isinstance(sub_ast, (bool, int, float)):
                return
            for a in sub_ast.args:
                visit_node(a)

    return visit_node(ast_node)


def move_constants_before_loop(ast_node):
    """Moves :class:`pystencils.ast.SympyAssignment` nodes out of loop body if they are iteration independent.

    Call this after creating the loop structure with :func:`make_loop_over_domain`
    """
    def find_block_to_move_to(node):
        """
        Traverses parents of node as long as the symbols are independent and returns a (parent) block
        the assignment can be safely moved to
        :param node: SympyAssignment inside a Block
        :return blockToInsertTo, childOfBlockToInsertBefore
        """
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
                break
            else:
                critical_symbols = set([s.name for s in element.symbols_defined])
            if set([s.name for s in node.undefined_symbols]).intersection(critical_symbols):
                break
            prev_element = element
            element = element.parent
        return last_block, last_block_child

    def check_if_assignment_already_in_block(assignment, target_block, rhs_or_lhs=True):
        for arg in target_block.args:
            if type(arg) is not ast.SympyAssignment:
                continue
            if (rhs_or_lhs and arg.rhs == assignment.rhs) or (not rhs_or_lhs and arg.lhs == assignment.lhs):
                return arg
        return None

    def get_blocks(node, result_list):
        if isinstance(node, ast.Block):
            result_list.append(node)
        if isinstance(node, ast.Node):
            for a in node.args:
                get_blocks(a, result_list)

    all_blocks = []
    get_blocks(ast_node, all_blocks)
    for block in all_blocks:
        children = block.take_child_nodes()
        for child in children:

            if not isinstance(child, ast.SympyAssignment):  # only move SympyAssignments
                block.append(child)
                continue

            target, child_to_insert_before = find_block_to_move_to(child)
            if target == block:     # movement not possible
                target.append(child)
            else:
                if isinstance(child, ast.SympyAssignment):
                    exists_already = check_if_assignment_already_in_block(child, target, False)
                else:
                    exists_already = False

                if not exists_already:
                    target.insert_before(child, child_to_insert_before)
                elif exists_already and exists_already.rhs == child.rhs:
                    if target.args.index(exists_already) > target.args.index(child_to_insert_before):
                        assert target.args.count(exists_already) == 1
                        assert target.args.count(child_to_insert_before) == 1
                        target.args.remove(exists_already)
                        target.insert_before(exists_already, child_to_insert_before)
                else:
                    # this variable already exists in outer block, but with different rhs
                    # -> symbol has to be renamed
                    assert isinstance(child.lhs, TypedSymbol)
                    new_symbol = TypedSymbol(sp.Dummy().name, child.lhs.dtype)
                    target.insert_before(ast.SympyAssignment(new_symbol, child.rhs, is_const=child.is_const),
                                         child_to_insert_before)
                    block.append(ast.SympyAssignment(child.lhs, new_symbol, is_const=child.is_const))


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
    inner_loop = [loop for loop in all_loops if loop.is_innermost_loop]
    assert len(inner_loop) == 1, "Error in AST: multiple innermost loops. Was split transformation already called?"
    inner_loop = inner_loop[0]
    assert type(inner_loop.body) is ast.Block
    outer_loop = [loop for loop in all_loops if loop.is_outermost_loop]
    assert len(outer_loop) == 1, "Error in AST, multiple outermost loops."
    outer_loop = outer_loop[0]

    symbols_with_temporary_array = OrderedDict()
    assignment_map = OrderedDict((a.lhs, a) for a in inner_loop.body.args if hasattr(a, 'lhs'))

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
                    if not isinstance(new_symbol, Field.Access) and \
                            new_symbol not in symbols_with_temporary_array:
                        symbols_to_process.append(new_symbol)
            symbols_resolved.add(s)

        for symbol in symbol_group:
            if not isinstance(symbol, Field.Access):
                assert type(symbol) is TypedSymbol
                new_ts = TypedSymbol(symbol.name, PointerType(symbol.dtype))
                symbols_with_temporary_array[symbol] = sp.IndexedBase(
                    new_ts, shape=(1, ))[inner_loop.loop_counter_symbol]

        assignment_group = []
        for assignment in inner_loop.body.args:
            if assignment.lhs in symbols_resolved:
                # use fast_subs here because it checks if multiplications should be evaluated or not
                new_rhs = fast_subs(assignment.rhs, symbols_with_temporary_array)
                if not isinstance(assignment.lhs, Field.Access) and assignment.lhs in symbol_group:
                    assert type(assignment.lhs) is TypedSymbol
                    new_ts = TypedSymbol(assignment.lhs.name, PointerType(assignment.lhs.dtype))
                    new_lhs = sp.IndexedBase(new_ts, shape=(1, ))[inner_loop.loop_counter_symbol]
                else:
                    new_lhs = assignment.lhs
                assignment_group.append(ast.SympyAssignment(new_lhs, new_rhs))
        assignment_groups.append(assignment_group)

    new_loops = [
        inner_loop.new_loop_with_different_body(ast.Block(group))
        for group in assignment_groups
    ]
    inner_loop.parent.replace(inner_loop, ast.Block(new_loops))

    for tmp_array in symbols_with_temporary_array:
        tmp_array_pointer = TypedSymbol(tmp_array.name, PointerType(tmp_array.dtype))
        alloc_node = ast.TemporaryMemoryAllocation(tmp_array_pointer, inner_loop.stop, inner_loop.start)
        free_node = ast.TemporaryMemoryFree(alloc_node)
        outer_loop.parent.insert_front(alloc_node)
        outer_loop.parent.append(free_node)


def cut_loop(loop_node, cutting_points):
    """Cuts loop at given cutting points.

    One loop is transformed into len(cuttingPoints)+1 new loops that range from
    old_begin to cutting_points[1], ..., cutting_points[-1] to old_end

    Modifies the ast in place. Note Issue #5783 of SymPy. Deepcopy will evaluate mul
    https://github.com/sympy/sympy/issues/5783

    Returns:
        list of new loop nodes
    """
    if loop_node.step != 1:
        raise NotImplementedError("Can only split loops that have a step of 1")
    new_loops = ast.Block([])
    new_start = loop_node.start
    cutting_points = list(cutting_points) + [loop_node.stop]
    for new_end in cutting_points:
        if new_end - new_start == 1:
            new_body = deepcopy(loop_node.body)
            new_body.subs({loop_node.loop_counter_symbol: new_start})
            new_loops.append(new_body)
        elif new_end - new_start == 0:
            pass
        else:
            new_loop = ast.LoopOverCoordinate(
                deepcopy(loop_node.body), loop_node.coordinate_to_loop_over,
                new_start, new_end, loop_node.step)
            new_loops.append(new_loop)
        new_start = new_end
    loop_node.parent.replace(loop_node, new_loops)
    return new_loops


def simplify_conditionals(node: ast.Node, loop_counter_simplification: bool = False) -> None:
    """Removes conditionals that are always true/false.

    Args:
        node: ast node, all descendants of this node are simplified
        loop_counter_simplification: if enabled, tries to detect if a conditional is always true/false
                                     depending on the surrounding loop. For example if the surrounding loop goes from
                                     x=0 to 10 and the condition is x < 0, it is removed.
                                     This analysis needs the integer set library (ISL) islpy, so it is not done by
                                     default.
    """
    from sympy.codegen.rewriting import ReplaceOptim, optimize
    remove_casts = ReplaceOptim(lambda e: isinstance(e, CastFunc), lambda p: p.expr)

    for conditional in node.atoms(ast.Conditional):
        # TODO simplify conditional before the type system! Casts make it very hard here
        condition_expression = optimize(conditional.condition_expr, [remove_casts])
        condition_expression = sp.simplify(condition_expression)
        if condition_expression == sp.true:
            conditional.parent.replace(conditional, [conditional.true_block])
        elif condition_expression == sp.false:
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


def remove_conditionals_in_staggered_kernel(function_node: ast.KernelFunction, include_first=True) -> None:
    """Removes conditionals of a kernel that iterates over staggered positions by splitting the loops at last or
       first and last element"""

    all_inner_loops = [l for l in function_node.atoms(ast.LoopOverCoordinate) if l.is_innermost_loop]
    assert len(all_inner_loops) == 1, "Transformation works only on kernels with exactly one inner loop"
    inner_loop = all_inner_loops.pop()

    for loop in parents_of_type(inner_loop, ast.LoopOverCoordinate, include_current=True):
        if include_first:
            cut_loop(loop, [loop.start + 1, loop.stop - 1])
        else:
            cut_loop(loop, [loop.stop - 1])

    simplify_conditionals(function_node.body, loop_counter_simplification=True)
    cleanup_blocks(function_node.body)

    move_constants_before_loop(function_node.body)
    cleanup_blocks(function_node.body)


# --------------------------------------- Helper Functions -------------------------------------------------------------
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
            raise ValueError(
                "All fields have to have the same number of spatial dimensions. Spatial field dimensions: "
                + str({f.name: f.spatial_shape
                       for f in fields}))

    layouts = set([field.layout for field in fields])
    if len(layouts) > 1:
        raise ValueError(
            "Due to different layout of the fields no optimal loop ordering exists "
            + str({f.name: f.layout
                   for f in fields}))
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


def get_loop_counter_symbol_hierarchy(ast_node):
    """Determines the loop counter symbols around a given AST node.
    :param ast_node: the AST node
    :return: list of loop counter symbols, where the first list entry is the symbol of the innermost loop
    """
    result = []
    node = ast_node
    while node is not None:
        node = get_next_parent_of_type(node, ast.LoopOverCoordinate)
        if node:
            result.append(node.loop_counter_symbol)
    return result


def replace_inner_stride_with_one(ast_node: ast.KernelFunction) -> None:
    """Replaces the stride of the innermost loop of a variable sized kernel with 1 (assumes optimal loop ordering).

    Variable sized kernels can handle arbitrary field sizes and field shapes. However, the kernel is most efficient
    if the innermost loop accesses the fields with stride 1. The inner loop can also only be vectorized if the inner
    stride is 1. This transformation hard codes this inner stride to one to enable e.g. vectorization.

    Warning: the assumption is not checked at runtime!
    """
    inner_loops = []
    inner_loop_counters = set()
    for loop in filtered_tree_iteration(ast_node,
                                        ast.LoopOverCoordinate,
                                        stop_type=ast.SympyAssignment):
        if loop.is_innermost_loop:
            inner_loops.append(loop)
            inner_loop_counters.add(loop.coordinate_to_loop_over)

    if len(inner_loop_counters) != 1:
        raise ValueError("Inner loops iterate over different coordinates")

    inner_loop_counter = inner_loop_counters.pop()

    parameters = ast_node.get_parameters()
    stride_params = [
        p.symbol for p in parameters
        if p.is_field_stride and p.symbol.coordinate == inner_loop_counter
    ]
    subs_dict = {stride_param: 1 for stride_param in stride_params}
    if subs_dict:
        ast_node.subs(subs_dict)


def loop_blocking(ast_node: ast.KernelFunction, block_size) -> int:
    """Blocking of loops to enhance cache locality. Modifies the ast node in-place.

    Args:
        ast_node: kernel function node before vectorization transformation has been applied
        block_size: sequence defining block size in x, y, (z) direction.
                    If chosen as zero the direction will not be used for blocking.

    Returns:
        number of dimensions blocked
    """
    loops = [
        l for l in filtered_tree_iteration(
            ast_node, ast.LoopOverCoordinate, stop_type=ast.SympyAssignment)
    ]
    body = ast_node.body

    coordinates = []
    coordinates_taken_into_account = 0
    loop_starts = {}
    loop_stops = {}

    for loop in loops:
        coord = loop.coordinate_to_loop_over
        if coord not in coordinates:
            coordinates.append(coord)
            loop_starts[coord] = loop.start
            loop_stops[coord] = loop.stop
        else:
            assert loop.start == loop_starts[coord] and loop.stop == loop_stops[coord], \
                f"Multiple loops over coordinate {coord} with different loop bounds"

    # Create the outer loops that iterate over the blocks
    outer_loop = None
    for coord in reversed(coordinates):
        if block_size[coord] == 0:
            continue
        coordinates_taken_into_account += 1
        body = ast.Block([outer_loop]) if outer_loop else body
        outer_loop = ast.LoopOverCoordinate(body,
                                            coord,
                                            loop_starts[coord],
                                            loop_stops[coord],
                                            step=block_size[coord],
                                            is_block_loop=True)

    ast_node.body = ast.Block([outer_loop])

    # modify the existing loops to only iterate within one block
    for inner_loop in loops:
        coord = inner_loop.coordinate_to_loop_over
        if block_size[coord] == 0:
            continue
        block_ctr = ast.LoopOverCoordinate.get_block_loop_counter_symbol(coord)
        loop_range = inner_loop.stop - inner_loop.start
        if sp.sympify(
                loop_range).is_number and loop_range % block_size[coord] == 0:
            stop = block_ctr + block_size[coord]
        else:
            stop = sp.Min(inner_loop.stop, block_ctr + block_size[coord])
        inner_loop.start = block_ctr
        inner_loop.stop = stop
    return coordinates_taken_into_account
