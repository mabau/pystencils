from collections import defaultdict
from functools import partial
from typing import Tuple, Union, Sequence

import numpy as np
import sympy as sp
# from pystencils.typing.leaf_typing import TypeAdder  # TODO this should be leaf_typing
from sympy.logic.boolalg import Boolean, BooleanFunction

import pystencils
from pystencils.cache import memorycache_if_hashable
from pystencils.typing.types import BasicType, VectorType, PointerType, create_type
from pystencils.typing.cast_functions import CastFunc, PointerArithmeticFunc
from pystencils.typing.typed_sympy import TypedSymbol


def typed_symbols(names, dtype, *args):
    # TODO docs, type hints
    symbols = sp.symbols(names, *args)
    if isinstance(symbols, Tuple):
        return tuple(TypedSymbol(str(s), dtype) for s in symbols)
    else:
        return TypedSymbol(str(symbols), dtype)


# noinspection PyPep8Naming
class address_of(sp.Function):
    # DONE: ask Martin
    # TODO: docstring
    # this is '&' in C
    is_Atom = True

    def __new__(cls, arg):
        obj = sp.Function.__new__(cls, arg)
        return obj

    @property
    def canonical(self):
        if hasattr(self.args[0], 'canonical'):
            return self.args[0].canonical
        else:
            raise NotImplementedError()

    @property
    def is_commutative(self):
        return self.args[0].is_commutative

    @property
    def dtype(self):
        if hasattr(self.args[0], 'dtype'):
            return PointerType(self.args[0].dtype, restrict=True)
        else:
            return PointerType('void', restrict=True)  # TODO this shouldn't work??? FIX: Allow BasicType to be Void and use that. Or raise exception


def get_base_type(data_type):
    # TODO: WTF is this?? DOCS!!!
    # TODO: This is unsafe.
    # TODO: remove
    # Pointer(Pointer(int))
    while data_type.base_type is not None:
        data_type = data_type.base_type
    return data_type


def peel_off_type(dtype, type_to_peel_off):
    # TODO: WTF is this??? DOCS!!!
    # TODO: used only once.... can be a lambda there
    while type(dtype) is type_to_peel_off:
        dtype = dtype.base_type
    return dtype


############################# This is basically our type system ########################################################

def result_type(*args: np.dtype):
    s = sorted(args, key=lambda x: x.itemsize)

    def kind_to_value(kind: str) -> int:
        if kind == 'f':
            return 3
        elif kind == 'i':
            return 2
        elif kind == 'u':
            return 1
        elif kind == 'b':
            return 0
        else:
            raise NotImplementedError(f'{kind=} is not a supported kind of a type. See "numpy.dtype.kind" for options')
    s = sorted(s, key=lambda x: kind_to_value(x.kind))
    return s[-1]


def collate_types(types: Sequence[Union[BasicType, VectorType]]):
    """
    Takes a sequence of types and returns their "common type" e.g. (float, double, float) -> double
    Uses the collation rules from numpy.
    """
    # # Pointer arithmetic case i.e. pointer + integer is allowed
    # if any(type(t) is PointerType for t in types):
    #     pointer_type = None
    #     for t in types:
    #         if type(t) is PointerType:
    #             if pointer_type is not None:
    #                 raise ValueError("Cannot collate the combination of two pointer types")
    #             pointer_type = t
    #         elif type(t) is BasicType:
    #             if not (t.is_int() or t.is_uint()):
    #                 raise ValueError("Invalid pointer arithmetic")
    #         else:
    #             raise ValueError("Invalid pointer arithmetic")
    #     return pointer_type
    #
    # # peel of vector types, if at least one vector type occurred the result will also be the vector type
    vector_type = [t for t in types if isinstance(t, VectorType)]
    # if not all_equal(t.width for t in vector_type):
    #     raise ValueError("Collation failed because of vector types with different width")
    # types = [peel_off_type(t, VectorType) for t in types]

    # now we should have a list of basic types - struct types are not yet supported
    assert all(type(t) is BasicType for t in types)

    result_numpy_type = result_type(*(t.numpy_dtype for t in types))
    result = BasicType(result_numpy_type)
    if vector_type:
        raise NotImplementedError("Vector type not implemented at the moment")
    #     result = VectorType(result, vector_type[0].width)
    return result


@memorycache_if_hashable(maxsize=2048)
def get_type_of_expression(expr,
                           default_float_type='double',
                           # TODO: we shouldn't need to have default. AST leaves should have a type
                           default_int_type='int',
                           # TODO: we shouldn't need to have default. AST leaves should have a type
                           symbol_type_dict=None):  # TODO: we shouldn't need to have default. AST leaves should have a type
    from pystencils.astnodes import ResolvedFieldAccess
    from pystencils.cpu.vectorization import vec_all, vec_any

    if default_float_type == 'float':
        default_float_type = 'float32'

    if not symbol_type_dict:
        symbol_type_dict = defaultdict(lambda: create_type('double'))

    get_type = partial(get_type_of_expression,
                       default_float_type=default_float_type,
                       default_int_type=default_int_type,
                       symbol_type_dict=symbol_type_dict)

    expr = sp.sympify(expr)
    if isinstance(expr, sp.Integer):
        return create_type(default_int_type)
    elif expr.is_real is False:
        return create_type((np.zeros((1,), default_float_type) * 1j).dtype)
    elif isinstance(expr, sp.Rational) or isinstance(expr, sp.Float):
        return create_type(default_float_type)
    elif isinstance(expr, ResolvedFieldAccess):
        return expr.field.dtype
    elif isinstance(expr, pystencils.field.Field.Access):
        return expr.field.dtype
    elif isinstance(expr, TypedSymbol):
        return expr.dtype
    elif isinstance(expr, sp.Symbol):
        # TODO delete if case
        if symbol_type_dict:
            return symbol_type_dict[expr.name]
        else:
            raise ValueError("All symbols inside this expression have to be typed! ", str(expr))
    elif isinstance(expr, CastFunc):
        return expr.args[1]
    elif isinstance(expr, (vec_any, vec_all)):
        return create_type("bool")
    elif hasattr(expr, 'func') and expr.func == sp.Piecewise:
        collated_result_type = collate_types(tuple(get_type(a[0]) for a in expr.args))
        collated_condition_type = collate_types(tuple(get_type(a[1]) for a in expr.args))
        if type(collated_condition_type) is VectorType and type(collated_result_type) is not VectorType:
            collated_result_type = VectorType(collated_result_type, width=collated_condition_type.width)
        return collated_result_type
    elif isinstance(expr, sp.Indexed):
        typed_symbol = expr.base.label
        return typed_symbol.dtype.base_type
    elif isinstance(expr, (Boolean, BooleanFunction)):
        # if any arg is of vector type return a vector boolean, else return a normal scalar boolean
        result = create_type("bool")
        vec_args = [get_type(a) for a in expr.args if isinstance(get_type(a), VectorType)]
        if vec_args:
            result = VectorType(result, width=vec_args[0].width)
        return result
    elif isinstance(expr, sp.Pow):
        base_type = get_type(expr.args[0])
        if expr.exp.is_integer:
            return base_type
        else:
            return collate_types([create_type(default_float_type), base_type])
    elif isinstance(expr, (sp.Sum, sp.Product)):
        return get_type(expr.args[0])
    elif isinstance(expr, sp.Expr):
        expr: sp.Expr
        if expr.args:
            types = tuple(get_type(a) for a in expr.args)
            # collate_types checks numpy_dtype in the special cases
            if any(not hasattr(t, 'numpy_dtype') for t in types):
                forbid_collation_to_complex = False
                forbid_collation_to_float = False
            else:
                forbid_collation_to_complex = expr.is_real is True
                forbid_collation_to_float = expr.is_integer is True
            return collate_types(
                types,
                forbid_collation_to_complex=forbid_collation_to_complex,
                forbid_collation_to_float=forbid_collation_to_float,
                default_float_type=default_float_type,
                default_int_type=default_int_type)
        else:
            if expr.is_integer:
                return create_type(default_int_type)
            else:
                return create_type(default_float_type)

    raise NotImplementedError("Could not determine type for", expr, type(expr))


############################# End This is basically our type system ##################################################


# TODO this seems quite wrong...
sympy_version = sp.__version__.split('.')
if int(sympy_version[0]) * 100 + int(sympy_version[1]) >= 109:
    # __setstate__ would bypass the contructor, so we remove it
    sp.Number.__getstate__ = sp.Basic.__getstate__
    del sp.Basic.__getstate__


    class FunctorWithStoredKwargs:
        def __init__(self, func, **kwargs):
            self.func = func
            self.kwargs = kwargs

        def __call__(self, *args):
            return self.func(*args, **self.kwargs)


    # __reduce_ex__ would strip kwargs, so we override it
    def basic_reduce_ex(self, protocol):
        if hasattr(self, '__getnewargs_ex__'):
            args, kwargs = self.__getnewargs_ex__()
        else:
            args, kwargs = self.__getnewargs__(), {}
        if hasattr(self, '__getstate__'):
            state = self.__getstate__()
        else:
            state = None
        return FunctorWithStoredKwargs(type(self), **kwargs), args, state


    sp.Number.__reduce_ex__ = sp.Basic.__reduce_ex__
    sp.Basic.__reduce_ex__ = basic_reduce_ex


def insert_casts(node):
    """Checks the types and inserts casts and pointer arithmetic where necessary.

    Args:
        node: the head node of the ast

    Returns:
        modified AST
    """
    from pystencils.astnodes import SympyAssignment, ResolvedFieldAccess, LoopOverCoordinate, Block

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
                casted_args.append(CastFunc(argument, target_dtype))
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
        return PointerArithmeticFunc(pointer, new_args)

    if isinstance(node, sp.AtomicExpr) or isinstance(node, CastFunc):
        return node
    args = []
    for arg in node.args:
        args.append(insert_casts(arg))
    # TODO indexed, LoopOverCoordinate
    if node.func in (sp.Add, sp.Mul, sp.Or, sp.And, sp.Pow, sp.Eq, sp.Ne, sp.Lt, sp.Le, sp.Gt, sp.Ge):
        # TODO optimize pow, don't cast integer on double
        types = [get_type_of_expression(arg) for arg in args]
        assert len(types) > 0
        # Never ever, ever collate to float type for boolean functions!
        target = collate_types(types, forbid_collation_to_float=isinstance(node.func, BooleanFunction))
        zipped = list(zip(args, types))
        if target.func is PointerType:
            assert node.func is sp.Add
            return pointer_arithmetic(zipped)
        else:
            return node.func(*cast(zipped, target))
    elif node.func is SympyAssignment:
        lhs = args[0]
        rhs = args[1]
        target = get_type_of_expression(lhs)
        if target.func is PointerType:
            return node.func(*args)  # TODO fix, not complete
        else:
            return node.func(lhs, *cast([(rhs, get_type_of_expression(rhs))], target))
    elif node.func is ResolvedFieldAccess:
        return node
    elif node.func is Block:
        for old_arg, new_arg in zip(node.args, args):
            node.replace(old_arg, new_arg)
        return node
    elif node.func is LoopOverCoordinate:
        for old_arg, new_arg in zip(node.args, args):
            node.replace(old_arg, new_arg)
        return node
    elif node.func is sp.Piecewise:
        expressions = [expr for (expr, _) in args]
        types = [get_type_of_expression(expr) for expr in expressions]
        target = collate_types(types)
        zipped = list(zip(expressions, types))
        casted_expressions = cast(zipped, target)
        args = [
            arg.func(*[expr, arg.cond])
            for (arg, expr) in zip(args, casted_expressions)
        ]

    return node.func(*args)


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
