from collections import namedtuple
from typing import Union, Tuple, Any, DefaultDict
import logging

import numpy as np

import sympy as sp
from sympy import Piecewise
from sympy.core.relational import Relational
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.functions.elementary.trigonometric import TrigonometricFunction, InverseTrigonometricFunction
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.codegen import Assignment
from sympy.logic.boolalg import BooleanFunction
from sympy.logic.boolalg import BooleanAtom

from pystencils import astnodes as ast
from pystencils.functions import DivFunc, AddressOf
from pystencils.cpu.vectorization import vec_all, vec_any
from pystencils.field import Field
from pystencils.typing.types import BasicType, PointerType
from pystencils.typing.utilities import collate_types
from pystencils.typing.cast_functions import CastFunc, BooleanCastFunc
from pystencils.typing.typed_sympy import TypedSymbol
from pystencils.fast_approximation import fast_sqrt, fast_division, fast_inv_sqrt
from pystencils.utils import ContextVar


class TypeAdder:
    # TODO: specification -> jupyter notebook
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

    def __init__(self, type_for_symbol: DefaultDict[str, BasicType], default_number_float: BasicType,
                 default_number_int: BasicType):
        self.type_for_symbol = type_for_symbol
        self.default_number_float = ContextVar(default_number_float)
        self.default_number_int = ContextVar(default_number_int)

    def visit(self, obj):
        if isinstance(obj, (list, tuple)):
            return [self.visit(e) for e in obj]
        if isinstance(obj, (sp.Eq, ast.SympyAssignment, Assignment)):
            return self.process_assignment(obj)
        elif isinstance(obj, ast.Conditional):
            condition, condition_type = self.figure_out_type(obj.condition_expr)
            assert condition_type == BasicType('bool')
            true_block = self.visit(obj.true_block)
            false_block = None if obj.false_block is None else self.visit(
                obj.false_block)
            return ast.Conditional(condition, true_block=true_block, false_block=false_block)
        elif isinstance(obj, ast.Block):
            return ast.Block([self.visit(e) for e in obj.args])
        elif isinstance(obj, ast.Node) and not isinstance(obj, ast.LoopOverCoordinate):
            return obj
        else:
            raise ValueError("Invalid object in kernel " + str(type(obj)))

    def process_assignment(self, assignment: Union[sp.Eq, ast.SympyAssignment, Assignment]) -> ast.SympyAssignment:
        # for checks it is crucial to process rhs before lhs to catch e.g. a = a + 1
        new_rhs, rhs_type = self.figure_out_type(assignment.rhs)

        lhs = assignment.lhs
        if not isinstance(lhs, (Field.Access, TypedSymbol)):
            if isinstance(lhs, sp.Symbol):
                self.type_for_symbol[lhs.name] = rhs_type
            else:
                raise ValueError(f'Lhs: `{lhs}` is not a subtype of sp.Symbol')
        new_lhs, lhs_type = self.figure_out_type(lhs)
        assert isinstance(new_lhs, (Field.Access, TypedSymbol))

        if lhs_type != rhs_type:
            logging.warning(f'Lhs"{new_lhs} of type "{lhs_type}" is assigned with a different datatype '
                            f'rhs: "{new_rhs}" of type "{rhs_type}".')
            return ast.SympyAssignment(new_lhs, CastFunc(new_rhs, lhs_type))
        else:
            return ast.SympyAssignment(new_lhs, new_rhs)

    # Type System Specification
    # - Defined Types: TypedSymbol, Field, Field.Access, ...?
    # - Indexed: always unsigned_integer64
    # - Undefined Types: Symbol
    #       - Is specified in Config in the dict or as 'default_type' or behaves like `auto` in the case of lhs.
    # - Constants/Numbers: Are either integer or floating. The precision and sign is specified via config
    #       - Example: 1.4 config:float32 -> float32
    # - Expressions deduce types from arguments
    # - Functions deduce types from arguments
    # - default_type and default_float and default_int can be given for a list of assignment, or
    #   individually as a list for assignment

    # Possible Problems - Do we need to support this?
    # - Mixture in expression with int and float
    # - Mixture in expression with uint64 and sint64
    # TODO Logging: Lowest log level should log all casts ----> cast factory, make cast should contain logging
    def figure_out_type(self, expr) -> Tuple[Any, Union[BasicType, PointerType]]:
        # Trivial cases
        from pystencils.field import Field
        import pystencils.integer_functions
        from pystencils.bit_masks import flag_cond
        bool_type = BasicType('bool')

        # TOOO: check the access
        if isinstance(expr, Field.Access):
            return expr, expr.dtype
        elif isinstance(expr, TypedSymbol):
            return expr, expr.dtype
        elif isinstance(expr, sp.Symbol):
            t = TypedSymbol(expr.name, self.type_for_symbol[expr.name])
            return t, t.dtype
        elif isinstance(expr, np.generic):
            assert False, f'Why do we have a np.generic in rhs???? {expr}'
        elif isinstance(expr, (sp.core.numbers.Infinity, sp.core.numbers.NegativeInfinity)):
            return expr, BasicType('float32')  # see https://en.cppreference.com/w/cpp/numeric/math/INFINITY
        elif isinstance(expr, sp.Number):
            if expr.is_Integer:
                data_type = self.default_number_int.get()
            elif expr.is_Float or expr.is_Rational:
                data_type = self.default_number_float.get()
            else:
                assert False, f'{sp.Number} is neither Float nor Integer'
            return CastFunc(expr, data_type), data_type
        elif isinstance(expr, AddressOf):
            of = expr.args[0]
            # TODO Basically this should do address_of already
            assert isinstance(of, (Field.Access, TypedSymbol, Field))
            return expr, PointerType(of.dtype)
        elif isinstance(expr, BooleanAtom):
            return expr, bool_type
        elif isinstance(expr, Relational):
            # TODO Jan: Code duplication with general case
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
        elif isinstance(expr, ast.ConditionalFieldAccess):
            access, access_type = self.figure_out_type(expr.access)
            value, value_type = self.figure_out_type(expr.outofbounds_value)
            condition, condition_type = self.figure_out_type(expr.outofbounds_condition)
            assert condition_type == bool_type
            collated_type = collate_types([access_type, value_type])
            if collated_type == access_type:
                new_access = access
            else:
                logging.warning(f"In {expr} the Field Access had to be casted to {collated_type}. This is "
                                f"probably due to a type missmatch of the Field and the value of "
                                f"ConditionalFieldAccess")
                new_access = CastFunc(access, collated_type)

            new_value = value if value_type == collated_type else CastFunc(value, collated_type)
            return expr.func(new_access, condition, new_value), collated_type
        elif isinstance(expr, (vec_any, vec_all)):
            return expr, bool_type
        elif isinstance(expr, BooleanFunction):
            args_types = [self.figure_out_type(a) for a in expr.args]
            new_args = [a if t.dtype_eq(bool_type) else BooleanCastFunc(a, bool_type) for a, t in args_types]
            return expr.func(*new_args), bool_type
        elif type(expr, ) in pystencils.integer_functions.__dict__.values():
            args_types = [self.figure_out_type(a) for a in expr.args]
            collated_type = collate_types([t for _, t in args_types])
            # TODO: should we downcast to integer? If yes then which integer type?
            if not collated_type.is_int():
                raise ValueError(f"Integer functions need to be used with integer types but {collated_type} was given")

            return expr, collated_type
        elif isinstance(expr, flag_cond):
            #   do not process the arguments to the bit shift - they must remain integers
            args_types = [self.figure_out_type(a) for a in (expr.args[i] for i in range(2, len(expr.args)))]
            collated_type = collate_types([t for _, t in args_types])
            new_expressions = [a if t.dtype_eq(collated_type) else CastFunc(a, collated_type) for a, t in args_types]
            return expr.func(expr.args[0], expr.args[1], *new_expressions), collated_type
        # elif isinstance(expr, sp.Mul):
        #    raise NotImplementedError('sp.Mul')
        #    # TODO can we ignore this and move it to general expr handling, i.e. removing Mul? (See todo in backend)
        #    # args_types = [self.figure_out_type(arg) for arg in expr.args if arg not in (-1, 1)]
        elif isinstance(expr, sp.Indexed):
            typed_symbol = expr.base.label
            return expr, typed_symbol.dtype
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
        elif isinstance(expr, (sp.Pow, sp.exp, InverseTrigonometricFunction, TrigonometricFunction,
                               HyperbolicFunction, sp.log)):
            args_types = [self.figure_out_type(arg) for arg in expr.args]
            collated_type = collate_types([t for _, t in args_types])
            new_args = [a if t.dtype_eq(collated_type) else CastFunc(a, collated_type) for a, t in args_types]
            new_func = expr.func(*new_args) if new_args else expr
            if collated_type == BasicType('float64'):
                return new_func, collated_type
            else:
                return CastFunc(new_func, collated_type), collated_type
        elif isinstance(expr, (fast_sqrt, fast_division, fast_inv_sqrt)):
            args_types = [self.figure_out_type(arg) for arg in expr.args]
            collated_type = BasicType('float32')
            new_args = [a if t.dtype_eq(collated_type) else CastFunc(a, collated_type) for a, t in args_types]
            new_func = expr.func(*new_args) if new_args else expr
            return CastFunc(new_func, collated_type), collated_type
        elif isinstance(expr, (sp.Add, sp.Mul, sp.Abs, sp.Min, sp.Max, DivFunc, sp.UnevaluatedExpr)):
            args_types = [self.figure_out_type(arg) for arg in expr.args]
            collated_type = collate_types([t for _, t in args_types])
            if isinstance(collated_type, PointerType):
                if isinstance(expr, sp.Add):
                    return expr.func(*[a for a, _ in args_types]), collated_type
                else:
                    raise NotImplementedError(f'Pointer Arithmetic is implemented only for Add, not {expr}')
            new_args = [a if t.dtype_eq(collated_type) else CastFunc(a, collated_type) for a, t in args_types]

            if isinstance(expr, (sp.Add, sp.Mul)):
                return expr.func(*new_args, evaluate=False) if new_args else expr, collated_type
            else:
                return expr.func(*new_args) if new_args else expr, collated_type
        else:
            raise NotImplementedError(f'expr {type(expr)}: {expr} unknown to typing')
