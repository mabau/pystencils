from collections import namedtuple, defaultdict
from copy import copy
from typing import Union, Dict, Tuple, Any
import logging

import numpy as np

import sympy as sp
from sympy import Piecewise
from sympy.core.relational import Relational
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.codegen import Assignment
from sympy.logic.boolalg import BooleanFunction
from sympy.logic.boolalg import BooleanAtom

from pystencils import astnodes as ast
from pystencils.field import Field
from pystencils.typing.types import AbstractType, BasicType, create_type
from pystencils.typing.utilities import get_type_of_expression, collate_types
from pystencils.typing.cast_functions import CastFunc, BooleanCastFunc
from pystencils.typing.typed_sympy import TypedSymbol
from pystencils.utils import ContextVar


class TypeAdder:
    # TODO: Logs
    # TODO: specification
    # TODO: split this into checker and leaf typing
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

    def __init__(self, type_for_symbol: Dict[str, BasicType], default_number_float: BasicType,
                 default_number_int: BasicType):
        self.type_for_symbol = ContextVar(type_for_symbol)
        self.default_number_float = ContextVar(default_number_float)
        self.default_number_int = ContextVar(default_number_int)

    # TODO: check if this adds only types to leave nodes of AST, get type info
    def visit(self, obj):

        if isinstance(obj, (list, tuple)):
            return [self.visit(e) for e in obj]
        if isinstance(obj, (sp.Eq, ast.SympyAssignment, Assignment)):
            return self.process_assignment(obj)
        elif isinstance(obj, ast.Conditional):
            false_block = None if obj.false_block is None else self.visit(
                obj.false_block)
            result = ast.Conditional(self.process_expression(
                obj.condition_expr, type_constants=False),
                true_block=self.visit(obj.true_block),
                false_block=false_block)
            return result
        elif isinstance(obj, ast.Block):
            result = ast.Block([self.visit(e) for e in obj.args])
            return result
        elif isinstance(obj, ast.Node) and not isinstance(obj, ast.LoopOverCoordinate):
            return obj
        else:
            raise ValueError("Invalid object in kernel " + str(type(obj)))

    def process_assignment(self, assignment: Union[sp.Eq, ast.SympyAssignment, Assignment]) -> ast.SympyAssignment:
        # for checks it is crucial to process rhs before lhs to catch e.g. a = a + 1
        new_rhs, rhs_type = self.figure_out_type(assignment.rhs)
        # TODO:
        dt = copy(rhs_type)  # The copy is necessary because BasicType has sympy shinanigans
        dd = defaultdict(lambda: BasicType(dt))
        dd.update(self.type_for_symbol.get())
        with self.type_for_symbol(dd):
            new_lhs, lhs_type = self.figure_out_type(assignment.lhs)
        # TODO add symbol to dict with type if defined!
        if lhs_type != rhs_type:
            logging.warning(f'Lhs"{new_lhs} of type "{lhs_type}" is assigned with a different datatype '
                            f'rhs: "{new_rhs}" of type "{rhs_type}".')
            return ast.SympyAssignment(new_lhs, CastFunc(new_rhs, lhs_type))
        else:
            return ast.SympyAssignment(new_lhs, new_rhs)

    # Type System Specification
    # - Defined Types: TypedSymbol, Field, Field.Access, ...?
    # - Indexed: always unsigned_integer64
    # - Undefined Types: Symbol - Is specified in Config in the dict or as 'default_type'
    # - Constants/Numbers: Are either integer or floating. The precision and sign is specified via config
    #       - Example: 1.4 config:float32 -> float32
    # - Expressions deduce types from arguments
    # - Functions deduce types from arguments
    # - default_type and default_float and default_int can be given for a list of assignment, or
    #   individually as a list for assignment

    # Possible Problems - Do we need to support this?
    # - Mixture in expression with int and float
    # - Mixture in expression with uint64 and sint64

    def figure_out_type(self, expr) -> Tuple[Any, BasicType]:  # TODO or abstract type? vector type?
        # Trivial cases
        from pystencils.field import Field
        import pystencils.integer_functions
        from pystencils.bit_masks import flag_cond
        bool_type = BasicType('bool')

        if isinstance(expr, Field.Access):
            return expr, expr.dtype
        elif isinstance(expr, TypedSymbol):
            return expr, expr.dtype
        elif isinstance(expr, sp.Symbol):
            t = TypedSymbol(expr.name, self.type_for_symbol.get()[expr.name])  # TODO with or without name
            return t, t.dtype
        elif isinstance(expr, np.generic):
            assert False, f'Why do we have a np.generic in rhs???? {expr}'
        elif isinstance(expr, sp.Number):
            if expr.is_Integer:
                data_type = self.default_number_int.get()
            elif expr.is_Float or expr.is_Rational:
                data_type = self.default_number_float.get()
            else:
                assert False, f'{sp.Number} is neither Float nor Integer'
            return CastFunc(expr, data_type), data_type
        elif isinstance(expr, BooleanAtom):
            return expr, bool_type
        elif isinstance(expr, Relational):
            # TODO JAN: Code duplication with general case
            args_types = [self.figure_out_type(arg) for arg in expr.args]
            collated_type = collate_types([t for _, t in args_types])
            if isinstance(expr, sp.Equality) and collated_type.is_float():
                logging.warning(f"Using floating point numbers in equality comparison: {expr}")
            new_args = [a if t.dtype_eq(collated_type) else CastFunc(a, collated_type) for a, t in args_types]
            new_eq = expr.func(*new_args)
            return new_eq, bool_type
        elif isinstance(expr, CastFunc):
            new_expr, _ = self.figure_out_type(expr.expr)
            return expr.func(*[new_expr, expr.dtype]), expr.dtype
        elif isinstance(expr, BooleanFunction):
            args_types = [self.figure_out_type(a) for a in expr.args]
            new_args = [a if t.dtype_eq(bool_type) else BooleanCastFunc(a, bool_type) for a, t in args_types]
            return expr.func(*new_args), bool_type
        elif type(expr, ) in pystencils.integer_functions.__dict__.values():
            raise NotImplementedError('integer_functions')
        elif isinstance(expr, flag_cond):
            #   do not process the arguments to the bit shift - they must remain integers
            raise NotImplementedError('flag_cond')
        #elif isinstance(expr, sp.Mul):
        #    raise NotImplementedError('sp.Mul')
        #    # TODO can we ignore this and move it to general expr handling, i.e. removing Mul?
        #    # args_types = [self.figure_out_type(arg) for arg in expr.args if arg not in (-1, 1)]
        elif isinstance(expr, sp.Indexed):
            raise NotImplementedError('sp.Indexed')
        elif isinstance(expr, sp.Pow):
            args_types = [self.figure_out_type(arg) for arg in expr.args]
            collated_type = collate_types([t for _, t in args_types])
            return expr.func(*[a for a, _ in args_types]), collated_type
        elif isinstance(expr, ExprCondPair):
            expr_expr, expr_type = self.figure_out_type(expr.expr)
            condition, condition_type = self.figure_out_type(expr.cond)
            if condition_type != bool_type:
                logging.warning(f'Condition "{condition}" is of type "{condition_type}" and not "bool"')
            return expr.func(expr_expr, condition), expr_type
        elif isinstance(expr, Piecewise):
            args_types = [self.figure_out_type(arg) for arg in expr.args]
            collated_type = collate_types([t for _, t in args_types])
            new_args = []
            for a, t in args_types:
                if t != collated_type:
                    if isinstance(a, ExprCondPair):
                        new_args.append(a.func(CastFunc(a.expr, collated_type), a.cond))
                    else:
                        new_args.append(CastFunc(a, collated_type))
                else:
                    new_args.append(a)
            return expr.func(*new_args) if new_args else expr, collated_type
        else:
            args_types = [self.figure_out_type(arg) for arg in expr.args]
            collated_type = collate_types([t for _, t in args_types])
            new_args = [a if t.dtype_eq(collated_type) else CastFunc(a, collated_type) for a, t in args_types]
            return expr.func(*new_args) if new_args else expr, collated_type

    def apply_type(self, expr, data_type: AbstractType):
        pass

    def process_expression(self, rhs, type_constants=True):  # TODO default_type as parameter
        import pystencils.integer_functions
        from pystencils.bit_masks import flag_cond

        if isinstance(rhs, Field.Access):
            return rhs
        elif isinstance(rhs, TypedSymbol):
            return rhs
        elif isinstance(rhs, sp.Symbol):
            return TypedSymbol(rhs.name, self._type_for_symbol[rhs.name])
        elif type_constants and isinstance(rhs, np.generic):
            assert False, f'Why do we have a np.generic in rhs???? {rhs}'
            # return CastFunc(rhs, create_type(rhs.dtype))
        elif type_constants and isinstance(rhs, sp.Number):
            return CastFunc(rhs, create_type(self._type_for_symbol['_constant']))
        # Very important that this clause comes before BooleanFunction
        elif isinstance(rhs, sp.Equality):
            if isinstance(rhs.args[1], sp.Number):
                return sp.Equality(
                    self.process_expression(rhs.args[0], type_constants),
                    rhs.args[1])  # TODO: process args[1] as number with a good type
            else:
                return sp.Equality(
                    self.process_expression(rhs.args[0], type_constants),
                    self.process_expression(rhs.args[1], type_constants))
        elif isinstance(rhs, CastFunc):
            return CastFunc(
                self.process_expression(rhs.args[0], type_constants=False),  # TODO: recommend type
                rhs.dtype)
        elif isinstance(rhs, BooleanFunction) or \
                type(rhs) in pystencils.integer_functions.__dict__.values():
            new_args = [self.process_expression(a, type_constants) for a in rhs.args]
            types_of_expressions = [get_type_of_expression(a) for a in new_args]
            arg_type = collate_types(types_of_expressions, forbid_collation_to_float=True)
            new_args = [a if not hasattr(a, 'dtype') or a.dtype == arg_type
                        else CastFunc(a, arg_type)
                        for a in new_args]
            return rhs.func(*new_args)
        elif isinstance(rhs, flag_cond):  # TODO
            #   do not process the arguments to the bit shift - they must remain integers
            processed_args = (self.process_expression(a) for a in rhs.args[2:])
            return flag_cond(rhs.args[0], rhs.args[1], *processed_args)
        elif isinstance(rhs, sp.Mul):
            new_args = [
                self.process_expression(arg, type_constants)
                if arg not in (-1, 1) else arg for arg in rhs.args
            ]
            return rhs.func(*new_args) if new_args else rhs
        elif isinstance(rhs, sp.Indexed):
            return rhs
        elif isinstance(rhs, sp.Pow):
            # don't process exponents -> they should remain integers  # TODO
            return sp.Pow(self.process_expression(rhs.args[0], type_constants), rhs.args[1])
        else:
            new_args = [self.process_expression(arg, type_constants) for arg in rhs.args]
            return rhs.func(*new_args) if new_args else rhs

    def process_lhs(self, lhs: Union[Field.Access, TypedSymbol, sp.Symbol]):
        if not isinstance(lhs, (Field.Access, TypedSymbol)):
            return TypedSymbol(lhs.name, self.type_for_symbol.get()[lhs.name])
        else:
            return lhs
