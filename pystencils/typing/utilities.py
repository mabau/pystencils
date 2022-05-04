from collections import defaultdict
from functools import partial
from typing import Tuple, Union, Sequence

import numpy as np
import sympy as sp
from sympy.logic.boolalg import Boolean, BooleanFunction

import pystencils
from pystencils.cache import memorycache_if_hashable
from pystencils.typing.types import BasicType, VectorType, PointerType, create_type
from pystencils.typing.cast_functions import CastFunc
from pystencils.typing.typed_sympy import TypedSymbol
from pystencils.utils import all_equal


def typed_symbols(names, dtype, **kwargs):
    """
    Creates TypedSymbols with the same functionality as sympy.symbols
    Args:
        names: See sympy.symbols
        dtype: The data type all symbols will have
        **kwargs: Key value arguments passed to sympy.symbols

    Returns:
        TypedSymbols
    """
    symbols = sp.symbols(names, **kwargs)
    if isinstance(symbols, Tuple):
        return tuple(TypedSymbol(str(s), dtype) for s in symbols)
    else:
        return TypedSymbol(str(symbols), dtype)


def get_base_type(data_type):
    """
    Returns the BasicType of a Pointer or a Vector
    """
    while data_type.base_type is not None:
        data_type = data_type.base_type
    return data_type


def result_type(*args: np.dtype):
    """Returns the type of the result if the np.dtype arguments would be collated.
    We can't use numpy functionality, because numpy casts don't behave exactly like C casts"""
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
    # Pointer arithmetic case i.e. pointer + [int, uint] is allowed
    if any(isinstance(t, PointerType) for t in types):
        pointer_type = None
        for t in types:
            if isinstance(t, PointerType):
                if pointer_type is not None:
                    raise ValueError(f'Cannot collate the combination of two pointer types "{pointer_type}" and "{t}"')
                pointer_type = t
            elif isinstance(t, BasicType):
                if not (t.is_int() or t.is_uint()):
                    raise ValueError("Invalid pointer arithmetic")
            else:
                raise ValueError("Invalid pointer arithmetic")
        return pointer_type

    # # peel of vector types, if at least one vector type occurred the result will also be the vector type
    vector_type = [t for t in types if isinstance(t, VectorType)]
    if not all_equal(t.width for t in vector_type):
        raise ValueError("Collation failed because of vector types with different width")

    # TODO: check if this is needed
    # def peel_off_type(dtype, type_to_peel_off):
    #     while type(dtype) is type_to_peel_off:
    #         dtype = dtype.base_type
    #     return dtype
    # types = [peel_off_type(t, VectorType) for t in types]

    types = [t.base_type if isinstance(t, VectorType) else t for t in types]

    # now we should have a list of basic types - struct types are not yet supported
    assert all(type(t) is BasicType for t in types)

    result_numpy_type = result_type(*(t.numpy_dtype for t in types))
    result = BasicType(result_numpy_type)
    if vector_type:
        result = VectorType(result, vector_type[0].width)
    return result


# TODO get_type_of_expression should be used after leaf_typing. So no defaults should be necessary
@memorycache_if_hashable(maxsize=2048)
def get_type_of_expression(expr,
                           default_float_type='double',
                           default_int_type='int',
                           symbol_type_dict=None):
    from pystencils.astnodes import ResolvedFieldAccess
    from pystencils.cpu.vectorization import vec_all, vec_any

    if default_float_type == 'float':
        default_float_type = 'float32'

    if not symbol_type_dict:
        symbol_type_dict = defaultdict(lambda: create_type('double'))

    # TODO this line is quite hard to understand, if possible simpl
    get_type = partial(get_type_of_expression,
                       default_float_type=default_float_type,
                       default_int_type=default_int_type,
                       symbol_type_dict=symbol_type_dict)

    expr = sp.sympify(expr)
    if isinstance(expr, sp.Integer):
        return create_type(default_int_type)
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
            return collate_types(types)
        else:
            if expr.is_integer:
                return create_type(default_int_type)
            else:
                return create_type(default_float_type)

    raise NotImplementedError("Could not determine type for", expr, type(expr))


# Fix for sympy versions from 1.9
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
