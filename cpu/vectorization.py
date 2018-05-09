import sympy as sp
import warnings
from pystencils.integer_functions import modulo_floor
from pystencils.sympyextensions import fast_subs
from pystencils.data_types import TypedSymbol, VectorType, get_type_of_expression, cast_func, collate_types, PointerType
import pystencils.astnodes as ast
from pystencils.transformations import cut_loop


def vectorize(ast_node, vector_width=4):
    vectorize_inner_loops_and_adapt_load_stores(ast_node, vector_width)
    insert_vector_casts(ast_node)


def vectorize_inner_loops_and_adapt_load_stores(ast_node, vector_width=4):
    """Goes over all innermost loops, changes increment to vector width and replaces field accesses by vector type."""
    inner_loops = [n for n in ast_node.atoms(ast.LoopOverCoordinate) if n.is_innermost_loop]

    for loop_node in inner_loops:
        loop_range = loop_node.stop - loop_node.start

        # cut off loop tail, that is not a multiple of four
        cutting_point = modulo_floor(loop_range, vector_width) + loop_node.start
        loop_nodes = cut_loop(loop_node, [cutting_point])
        assert len(loop_nodes) in (1, 2)  # 2 for main and tail loop, 1 if loop range divisible by vector width
        loop_node = loop_nodes[0]
        
        # Find all array accesses (indexed) that depend on the loop counter as offset
        loop_counter_symbol = ast.LoopOverCoordinate.get_loop_counter_symbol(loop_node.coordinate_to_loop_over)
        substitutions = {}
        successful = True
        for indexed in loop_node.atoms(sp.Indexed):
            base, index = indexed.args
            if loop_counter_symbol in index.atoms(sp.Symbol):
                loop_counter_is_offset = loop_counter_symbol not in (index - loop_counter_symbol).atoms()
                if not loop_counter_is_offset:
                    successful = False
                    break
                typed_symbol = base.label
                assert type(typed_symbol.dtype) is PointerType, \
                    "Type of access is {}, {}".format(typed_symbol.dtype, indexed)
                substitutions[indexed] = cast_func(indexed, VectorType(typed_symbol.dtype.base_type, vector_width))
        if not successful:
            warnings.warn("Could not vectorize loop because of non-consecutive memory access")
            continue

        loop_node.step = vector_width
        loop_node.subs(substitutions)


def insert_vector_casts(ast_node):
    """Inserts necessary casts from scalar values to vector values."""

    def visit_expr(expr):
        if expr.func in (sp.Add, sp.Mul) or (isinstance(expr, sp.Rel) and not expr.func == cast_func) or \
                isinstance(expr, sp.boolalg.BooleanFunction):
            new_args = [visit_expr(a) for a in expr.args]
            arg_types = [get_type_of_expression(a) for a in new_args]
            if not any(type(t) is VectorType for t in arg_types):
                return expr
            else:
                target_type = collate_types(arg_types)
                casted_args = [cast_func(a, target_type) if t != target_type else a
                               for a, t in zip(new_args, arg_types)]
                return expr.func(*casted_args)
        elif expr.func is sp.Pow:
            new_arg = visit_expr(expr.args[0])
            return expr.func(new_arg, expr.args[1])
        elif expr.func == sp.Piecewise:
            new_results = [visit_expr(a[0]) for a in expr.args]
            new_conditions = [visit_expr(a[1]) for a in expr.args]
            types_of_results = [get_type_of_expression(a) for a in new_results]
            types_of_conditions = [get_type_of_expression(a) for a in new_conditions]

            result_target_type = get_type_of_expression(expr)
            condition_target_type = collate_types(types_of_conditions)
            if type(condition_target_type) is VectorType and type(result_target_type) is not VectorType:
                result_target_type = VectorType(result_target_type, width=condition_target_type.width)

            casted_results = [cast_func(a, result_target_type) if t != result_target_type else a
                              for a, t in zip(new_results, types_of_results)]

            casted_conditions = [cast_func(a, condition_target_type)
                                 if t != condition_target_type and a is not True else a
                                 for a, t in zip(new_conditions, types_of_conditions)]

            return sp.Piecewise(*[(r, c) for r, c in zip(casted_results, casted_conditions)])
        else:
            return expr

    def visit_node(node, substitution_dict):
        substitution_dict = substitution_dict.copy()
        for arg in node.args:
            if isinstance(arg, ast.SympyAssignment):
                assignment = arg
                subs_expr = fast_subs(assignment.rhs, substitution_dict,
                                      skip=lambda e: isinstance(e, ast.ResolvedFieldAccess))
                assignment.rhs = visit_expr(subs_expr)
                rhs_type = get_type_of_expression(assignment.rhs)
                if isinstance(assignment.lhs, TypedSymbol):
                    lhs_type = assignment.lhs.dtype
                    if type(rhs_type) is VectorType and type(lhs_type) is not VectorType:
                        new_lhs_type = VectorType(lhs_type, rhs_type.width)
                        new_lhs = TypedSymbol(assignment.lhs.name, new_lhs_type)
                        substitution_dict[assignment.lhs] = new_lhs
                        assignment.lhs = new_lhs
                elif assignment.lhs.func == cast_func:
                    lhs_type = assignment.lhs.args[1]
                    if type(lhs_type) is VectorType and type(rhs_type) is not VectorType:
                        assignment.rhs = cast_func(assignment.rhs, lhs_type)
            else:
                visit_node(arg, substitution_dict)

    visit_node(ast_node, {})
