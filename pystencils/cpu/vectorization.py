import warnings
from typing import Container, Union

import numpy as np
import sympy as sp
from sympy.logic.boolalg import BooleanFunction, BooleanAtom

import pystencils.astnodes as ast
from pystencils.backends.simd_instruction_sets import get_supported_instruction_sets, get_vector_instruction_set
from pystencils.typing import ( BasicType, PointerType, TypedSymbol, VectorType, CastFunc, collate_types,
                                get_type_of_expression, VectorMemoryAccess)
from pystencils.fast_approximation import fast_division, fast_inv_sqrt, fast_sqrt
from pystencils.functions import DivFunc
from pystencils.field import Field
from pystencils.integer_functions import modulo_ceil, modulo_floor
from pystencils.sympyextensions import fast_subs
from pystencils.transformations import cut_loop, filtered_tree_iteration, replace_inner_stride_with_one


# noinspection PyPep8Naming
class vec_any(sp.Function):
    nargs = (1,)


# noinspection PyPep8Naming
class vec_all(sp.Function):
    nargs = (1,)


class NontemporalFence(ast.Node):
    def __init__(self):
        super(NontemporalFence, self).__init__(parent=None)

    @property
    def symbols_defined(self):
        return set()

    @property
    def undefined_symbols(self):
        return set()

    @property
    def args(self):
        return []

    def __eq__(self, other):
        return isinstance(other, NontemporalFence)


class CachelineSize(ast.Node):
    symbol = sp.Symbol("_clsize")
    mask_symbol = sp.Symbol("_clsize_mask")
    last_symbol = sp.Symbol("_cl_lastvec")
    
    def __init__(self):
        super(CachelineSize, self).__init__(parent=None)

    @property
    def symbols_defined(self):
        return {self.symbol, self.mask_symbol, self.last_symbol}

    @property
    def undefined_symbols(self):
        return set()

    @property
    def args(self):
        return []

    def __eq__(self, other):
        return isinstance(other, CachelineSize)

    def __hash__(self):
        return hash(self.symbol)


def vectorize(kernel_ast: ast.KernelFunction, instruction_set: str = 'best',
              assume_aligned: bool = False, nontemporal: Union[bool, Container[Union[str, Field]]] = False,
              assume_inner_stride_one: bool = False, assume_sufficient_line_padding: bool = True):
    """Explicit vectorization using SIMD vectorization via intrinsics.

    Args:
        kernel_ast: abstract syntax tree (KernelFunction node)
        instruction_set: one of the supported vector instruction sets, currently ('sse', 'avx' and 'avx512')
        assume_aligned: assume that the first inner cell of each line is aligned. If false, only unaligned-loads are
                        used. If true, some of the loads are assumed to be from aligned memory addresses.
                        For example if x is the fastest coordinate, the access to center can be fetched via an
                        aligned-load instruction, for the west or east accesses potentially slower unaligend-load
                        instructions have to be used.
        nontemporal: a container of fields or field names for which nontemporal (streaming) stores are used.
                     If true, nontemporal access instructions are used for all fields.
        assume_inner_stride_one: kernels with non-constant inner loop bound and strides can not be vectorized since
                                 the inner loop stride is a runtime variable and thus might not be always 1.
                                 If this parameter is set to true, the inner stride is assumed to be always one.
                                 This has to be ensured at runtime!
        assume_sufficient_line_padding: if True and assume_inner_stride_one, no tail loop is created but loop is
                                        extended by at most (vector_width-1) elements
                                        assumes that at the end of each line there is enough padding with dummy data
                                        depending on the access pattern there might be additional padding
                                        required at the end of the array
    """
    if instruction_set == 'best':
        if get_supported_instruction_sets():
            instruction_set = get_supported_instruction_sets()[-1]
        else:
            instruction_set = 'avx'
    if instruction_set is None:
        return

    all_fields = kernel_ast.fields_accessed
    if nontemporal is None or nontemporal is False:
        nontemporal = {}
    elif nontemporal is True:
        nontemporal = all_fields

    if assume_inner_stride_one:
        replace_inner_stride_with_one(kernel_ast)

    field_float_dtypes = set(f.dtype for f in all_fields if f.dtype.is_float())
    if len(field_float_dtypes) != 1:
        raise NotImplementedError("Cannot vectorize kernels that contain accesses "
                                  "to differently typed floating point fields")
    float_size = field_float_dtypes.pop().numpy_dtype.itemsize
    assert float_size in (8, 4)
    # TODO: future work allow mixed precision fields
    default_float_type = 'double' if float_size == 8 else 'float'
    vector_is = get_vector_instruction_set(default_float_type, instruction_set=instruction_set)
    vector_width = vector_is['width']
    kernel_ast.instruction_set = vector_is

    strided = 'storeS' in vector_is and 'loadS' in vector_is
    keep_loop_stop = '{loop_stop}' in vector_is['storeA' if assume_aligned else 'storeU']
    vectorize_inner_loops_and_adapt_load_stores(kernel_ast, vector_width, assume_aligned, nontemporal,
                                                strided, keep_loop_stop, assume_sufficient_line_padding,
                                                default_float_type)
    # is in vectorize_inner_loops_and_adapt_load_stores.. insert_vector_casts(kernel_ast, default_float_type)


def vectorize_inner_loops_and_adapt_load_stores(ast_node, vector_width, assume_aligned, nontemporal_fields,
                                                strided, keep_loop_stop, assume_sufficient_line_padding,
                                                default_float_type):
    """Goes over all innermost loops, changes increment to vector width and replaces field accesses by vector type."""
    all_loops = filtered_tree_iteration(ast_node, ast.LoopOverCoordinate, stop_type=ast.SympyAssignment)
    inner_loops = [n for n in all_loops if n.is_innermost_loop]
    zero_loop_counters = {l.loop_counter_symbol: 0 for l in all_loops}

    for loop_node in inner_loops:
        loop_range = loop_node.stop - loop_node.start

        # cut off loop tail, that is not a multiple of four
        if keep_loop_stop:
            pass
        elif assume_aligned and assume_sufficient_line_padding:
            loop_range = loop_node.stop - loop_node.start
            new_stop = loop_node.start + modulo_ceil(loop_range, vector_width)
            loop_node.stop = new_stop
        else:
            cutting_point = modulo_floor(loop_range, vector_width) + loop_node.start
            loop_nodes = [l for l in cut_loop(loop_node, [cutting_point]).args if isinstance(l, ast.LoopOverCoordinate)]
            assert len(loop_nodes) in (0, 1, 2)  # 2 for main and tail loop, 1 if loop range divisible by vector width
            if len(loop_nodes) == 0:
                continue
            loop_node = loop_nodes[0]
            # TODO loop_node is the vectorized one

        # Find all array accesses (indexed) that depend on the loop counter as offset
        loop_counter_symbol = ast.LoopOverCoordinate.get_loop_counter_symbol(loop_node.coordinate_to_loop_over)
        substitutions = {}
        successful = True
        for indexed in loop_node.atoms(sp.Indexed):
            base, index = indexed.args
            if loop_counter_symbol in index.atoms(sp.Symbol):
                loop_counter_is_offset = loop_counter_symbol not in (index - loop_counter_symbol).atoms()
                aligned_access = (index - loop_counter_symbol).subs(zero_loop_counters) == 0
                stride = sp.simplify(index.subs({loop_counter_symbol: loop_counter_symbol + 1}) - index)
                if not loop_counter_is_offset and (not strided or loop_counter_symbol in stride.atoms()):
                    successful = False
                    break
                typed_symbol = base.label
                assert type(typed_symbol.dtype) is PointerType, \
                    f"Type of access is {typed_symbol.dtype}, {indexed}"

                vec_type = VectorType(typed_symbol.dtype.base_type, vector_width)
                use_aligned_access = aligned_access and assume_aligned
                nontemporal = False
                if hasattr(indexed, 'field'):
                    nontemporal = (indexed.field in nontemporal_fields) or (indexed.field.name in nontemporal_fields)
                substitutions[indexed] = VectorMemoryAccess(indexed, vec_type, use_aligned_access, nontemporal, True,
                                                            stride if strided else 1)
                if nontemporal:
                    # insert NontemporalFence after the outermost loop
                    parent = loop_node.parent
                    while type(parent.parent.parent) is not ast.KernelFunction:
                        parent = parent.parent
                    parent.parent.insert_after(NontemporalFence(), parent, if_not_exists=True)
                    # insert CachelineSize at the beginning of the kernel
                    parent.parent.insert_front(CachelineSize(), if_not_exists=True)
        if not successful:
            warnings.warn("Could not vectorize loop because of non-consecutive memory access")
            continue

        loop_node.step = vector_width
        loop_node.subs(substitutions)
        vector_int_width = ast_node.instruction_set['intwidth']
        vector_loop_counter = CastFunc(loop_counter_symbol, VectorType(loop_counter_symbol.dtype, vector_int_width)) \
                              + CastFunc(tuple(range(vector_int_width if type(vector_int_width) is int else 2)),
                                         VectorType(loop_counter_symbol.dtype, vector_int_width))

        fast_subs(loop_node, {loop_counter_symbol: vector_loop_counter},
                  skip=lambda e: isinstance(e, ast.ResolvedFieldAccess) or isinstance(e, VectorMemoryAccess))

        mask_conditionals(loop_node)

        from pystencils.rng import RNGBase
        substitutions = {}
        for rng in loop_node.atoms(RNGBase):
            new_result_symbols = [TypedSymbol(s.name, VectorType(s.dtype, width=vector_width))
                                  for s in rng.result_symbols]
            substitutions.update({s[0]: s[1] for s in zip(rng.result_symbols, new_result_symbols)})
            rng._symbols_defined = set(new_result_symbols)
        fast_subs(loop_node, substitutions, skip=lambda e: isinstance(e, RNGBase))
        insert_vector_casts(loop_node, default_float_type, vector_width)


def mask_conditionals(loop_body):
    def visit_node(node, mask):
        if isinstance(node, ast.Conditional):
            cond = node.condition_expr
            skip = (loop_body.loop_counter_symbol not in cond.atoms(sp.Symbol)) or cond.func in (vec_all, vec_any)
            cond = True if skip else cond

            true_mask = sp.And(cond, mask)
            visit_node(node.true_block, true_mask)
            if node.false_block:
                false_mask = sp.And(sp.Not(node.condition_expr), mask)
                visit_node(node, false_mask)
            if not skip:
                node.condition_expr = vec_any(node.condition_expr)
        elif isinstance(node, ast.SympyAssignment):
            if mask is not True:
                s = {ma: VectorMemoryAccess(*ma.args[0:4], sp.And(mask, ma.args[4]), *ma.args[5:])
                     for ma in node.atoms(VectorMemoryAccess)}
                node.subs(s)
        else:
            for arg in node.args:
                visit_node(arg, mask)

    visit_node(loop_body, mask=True)


def insert_vector_casts(ast_node, default_float_type='double', vector_width=4):
    """Inserts necessary casts from scalar values to vector values."""

    handled_functions = (sp.Add, sp.Mul, fast_division, fast_sqrt, fast_inv_sqrt, vec_any, vec_all, DivFunc,
                         sp.UnevaluatedExpr)

    def visit_expr(expr, default_type='double'):  # TODO get rid of default_type
        if isinstance(expr, VectorMemoryAccess):
            return VectorMemoryAccess(*expr.args[0:4], visit_expr(expr.args[4], default_type), *expr.args[5:])
        elif isinstance(expr, CastFunc):
            cast_type = expr.args[1]
            arg = visit_expr(expr.args[0])
            assert cast_type in [BasicType('float32'), BasicType('float64')],\
                f'Vectorization cannot vectorize type {cast_type}'
            return expr.func(arg, VectorType(cast_type, vector_width))
        elif expr.func is sp.Abs and 'abs' not in ast_node.instruction_set:
            new_arg = visit_expr(expr.args[0], default_type)
            base_type = get_type_of_expression(expr.args[0]).base_type if type(expr.args[0]) is VectorMemoryAccess \
                else get_type_of_expression(expr.args[0])
            pw = sp.Piecewise((-new_arg, new_arg < base_type.numpy_dtype.type(0)),
                              (new_arg, True))
            return visit_expr(pw, default_type)
        elif expr.func in handled_functions or isinstance(expr, sp.Rel) or isinstance(expr, BooleanFunction):
            if expr.func is sp.Mul and expr.args[0] == -1:
                # special treatment for the unary minus: make sure that the -1 has the same type as the argument
                dtype = int
                for arg in expr.atoms(VectorMemoryAccess):
                    if arg.dtype.base_type.is_float():
                        dtype = arg.dtype.base_type.numpy_dtype.type
                for arg in expr.atoms(TypedSymbol):
                    if type(arg.dtype) is VectorType and arg.dtype.base_type.is_float():
                        dtype = arg.dtype.base_type.numpy_dtype.type
                if dtype is not int:
                    if dtype is np.float32:
                        default_type = 'float'
                    expr = sp.Mul(dtype(expr.args[0]), *expr.args[1:])
            new_args = [visit_expr(a, default_type) for a in expr.args]
            arg_types = [get_type_of_expression(a, default_float_type=default_type) for a in new_args]
            if not any(type(t) is VectorType for t in arg_types):
                return expr
            else:
                target_type = collate_types(arg_types)
                casted_args = [
                    CastFunc(a, target_type) if t != target_type and not isinstance(a, VectorMemoryAccess) else a
                    for a, t in zip(new_args, arg_types)]
                return expr.func(*casted_args)
        elif expr.func is sp.Pow:
            new_arg = visit_expr(expr.args[0], default_type)
            return expr.func(new_arg, expr.args[1])
        elif expr.func == sp.Piecewise:
            new_results = [visit_expr(a[0], default_type) for a in expr.args]
            new_conditions = [visit_expr(a[1], default_type) for a in expr.args]
            types_of_results = [get_type_of_expression(a) for a in new_results]
            types_of_conditions = [get_type_of_expression(a) for a in new_conditions]

            result_target_type = get_type_of_expression(expr)
            condition_target_type = collate_types(types_of_conditions)
            if type(condition_target_type) is VectorType and type(result_target_type) is not VectorType:
                result_target_type = VectorType(result_target_type, width=condition_target_type.width)
            if type(condition_target_type) is not VectorType and type(result_target_type) is VectorType:
                condition_target_type = VectorType(condition_target_type, width=result_target_type.width)

            casted_results = [CastFunc(a, result_target_type) if t != result_target_type else a
                              for a, t in zip(new_results, types_of_results)]

            casted_conditions = [CastFunc(a, condition_target_type)
                                 if t != condition_target_type and a is not True else a
                                 for a, t in zip(new_conditions, types_of_conditions)]

            return sp.Piecewise(*[(r, c) for r, c in zip(casted_results, casted_conditions)])
        elif isinstance(expr, (sp.Number, TypedSymbol, BooleanAtom)):
            return expr
        else:
            # TODO better error string
            raise NotImplementedError(f'Should I raise or should I return now? {type(expr)} {expr}')

    def visit_node(node, substitution_dict, default_type='double'):
        substitution_dict = substitution_dict.copy()
        for arg in node.args:
            if isinstance(arg, ast.SympyAssignment):
                # TODO only if not remainder loop (? if no VectorAccess then remainder loop)
                assignment = arg
                # If there is a remainder loop we do not vectorise it, thus lhs will indicate this
                # if isinstance(assignment.lhs, ast.ResolvedFieldAccess):
                    # continue
                subs_expr = fast_subs(assignment.rhs, substitution_dict,
                                      skip=lambda e: isinstance(e, ast.ResolvedFieldAccess))
                assignment.rhs = visit_expr(subs_expr, default_type)
                rhs_type = get_type_of_expression(assignment.rhs)
                if isinstance(assignment.lhs, TypedSymbol):
                    lhs_type = assignment.lhs.dtype
                    if type(rhs_type) is VectorType and type(lhs_type) is not VectorType:
                        new_lhs_type = VectorType(lhs_type, rhs_type.width)
                        new_lhs = TypedSymbol(assignment.lhs.name, new_lhs_type)
                        substitution_dict[assignment.lhs] = new_lhs
                        assignment.lhs = new_lhs
                elif isinstance(assignment.lhs, VectorMemoryAccess):
                    assignment.lhs = visit_expr(assignment.lhs, default_type)
            elif isinstance(arg, ast.Conditional):
                arg.condition_expr = fast_subs(arg.condition_expr, substitution_dict,
                                               skip=lambda e: isinstance(e, ast.ResolvedFieldAccess))
                arg.condition_expr = visit_expr(arg.condition_expr, default_type)
                visit_node(arg, substitution_dict, default_type)
            else:
                visit_node(arg, substitution_dict, default_type)

    visit_node(ast_node, {}, default_float_type)
