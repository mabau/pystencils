from __future__ import annotations

import sympy as sp
from enum import Enum, auto

from ..types import (
    PsType,
    PsNumericType,
    PsBoolType,
    create_type,
    UserTypeSpec
)


def is_loop_counter_symbol(symbol):
    from ..defaults import DEFAULTS

    try:
        return DEFAULTS.spatial_counters.index(symbol)
    except ValueError:
        return None


class DynamicType(Enum):
    """Dynamic data type that will be resolved during kernel creation"""

    NUMERIC_TYPE = auto()
    """Use the default numeric type set for the kernel"""

    INDEX_TYPE = auto()
    """Use the default index type set for the kernel.
    
    This is guaranteed to be an interger type.
    """


class TypeAtom(sp.Atom):
    """Wrapper around a type to disguise it as a SymPy atom."""

    def __new__(cls, *args, **kwargs):
        return sp.Basic.__new__(cls)

    def __init__(self, dtype: PsType | DynamicType) -> None:
        self._dtype = dtype

    def _sympystr(self, *args, **kwargs):
        return str(self._dtype)

    def get(self) -> PsType | DynamicType:
        return self._dtype

    def _hashable_content(self):
        return (self._dtype,)
    

def assumptions_from_dtype(dtype: PsType | DynamicType):
    """Derives SymPy assumptions from :class:`PsAbstractType`

    Args:
        dtype (PsAbstractType): a pystencils data type
    Returns:
        A dict of SymPy assumptions
    """
    assumptions = dict()

    match dtype:
        case DynamicType.INDEX_TYPE:
            assumptions.update({"integer": True, "real": True})
        case DynamicType.NUMERIC_TYPE:
            assumptions.update({"real": True})
        case PsNumericType():
            if dtype.is_int():
                assumptions.update({"integer": True})
            if dtype.is_uint():
                assumptions.update({"negative": False})
            if dtype.is_int() or dtype.is_float():
                assumptions.update({"real": True})

    return assumptions


class TypedSymbol(sp.Symbol):

    _dtype: PsType | DynamicType

    def __new__(cls, *args, **kwds):
        obj = TypedSymbol.__xnew_cached_(cls, *args, **kwds)
        return obj

    def __new_stage2__(
        cls, name: str, dtype: UserTypeSpec | DynamicType, **kwargs
    ):  # TODO does not match signature of sp.Symbol???
        # TODO: also Symbol should be allowed  ---> see sympy Variable
        if not isinstance(dtype, DynamicType):
            dtype = create_type(dtype)

        assumptions = assumptions_from_dtype(dtype)
        assumptions.update(kwargs)

        obj = super(TypedSymbol, cls).__xnew__(cls, name, **assumptions)
        obj._dtype = dtype

        return obj

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(sp.core.cacheit(__new_stage2__))

    @property
    def dtype(self) -> PsType | DynamicType:
        #   mypy: ignore
        return self._dtype

    def _hashable_content(self):
        #   mypy: ignore
        return super()._hashable_content(), hash(self._dtype)

    def __getnewargs__(self):
        return self.name, self.dtype

    def __getnewargs_ex__(self):
        return (self.name, self.dtype), self.assumptions0

    @property
    def canonical(self):
        return self

    @property
    def reversed(self):
        return self

    @property
    def headers(self) -> set[str]:
        return self.dtype.required_headers if isinstance(self.dtype, PsType) else set()


class CastFunc(sp.Function):
    """Use this function to introduce a static type cast into the output code.

    Usage: ``CastFunc(expr, target_type)`` becomes, in C code, ``(target_type) expr``.
    The ``target_type`` may be a valid pystencils type specification parsable by `create_type`,
    or a special value of the `DynamicType` enum.
    These dynamic types can be used to select the target type according to the code generation context.
    """

    @staticmethod
    def as_numeric(expr):
        return CastFunc(expr, DynamicType.NUMERIC_TYPE)

    @staticmethod
    def as_index(expr):
        return CastFunc(expr, DynamicType.INDEX_TYPE)

    is_Atom = True

    def __new__(cls, *args, **kwargs):
        if len(args) != 2:
            pass
        expr, dtype, *other_args = args

        # If we have two consecutive casts, throw the inner one away.
        # This optimisation is only available for simple casts. Thus the == is intended here!
        if expr.__class__ == CastFunc:
            expr = expr.args[0]

        if not isinstance(dtype, (TypeAtom)):
            if isinstance(dtype, DynamicType):
                dtype = TypeAtom(dtype)
            else:
                dtype = TypeAtom(create_type(dtype))

        # to work in conditions of sp.Piecewise cast_func has to be of type Boolean as well
        # however, a cast_function should only be a boolean if its argument is a boolean, otherwise this leads
        # to problems when for example comparing cast_func's for equality
        #
        # lhs = bitwise_and(a, cast_func(1, 'int'))
        # rhs = cast_func(0, 'int')
        # print( sp.Ne(lhs, rhs) ) # would give true if all cast_funcs are booleans
        # -> thus a separate class boolean_cast_func is introduced
        if isinstance(expr, sp.logic.boolalg.Boolean) and (
            not isinstance(expr, TypedSymbol) or isinstance(expr.dtype, PsBoolType)
        ):
            cls = BooleanCastFunc

        return sp.Function.__new__(cls, expr, dtype, *other_args, **kwargs)

    @property
    def canonical(self):
        if hasattr(self.args[0], "canonical"):
            return self.args[0].canonical
        else:
            raise NotImplementedError()

    @property
    def is_commutative(self):
        return self.args[0].is_commutative

    @property
    def dtype(self) -> PsType | DynamicType:
        assert isinstance(self.args[1], TypeAtom)
        return self.args[1].get()

    @property
    def expr(self):
        return self.args[0]

    @property
    def is_integer(self):
        if self.dtype == DynamicType.INDEX_TYPE:
            return True
        elif isinstance(self.dtype, PsNumericType):
            return self.dtype.is_int() or super().is_integer
        else:
            return super().is_integer

    @property
    def is_negative(self):
        """
        See :func:`.TypedSymbol.is_integer`
        """
        if isinstance(self.dtype, PsNumericType):
            if self.dtype.is_uint():
                return False

        return super().is_negative

    @property
    def is_nonnegative(self):
        """
        See :func:`.TypedSymbol.is_integer`
        """
        if self.is_negative is False:
            return True
        else:
            return super().is_nonnegative

    @property
    def is_real(self):
        """
        See :func:`.TypedSymbol.is_integer`
        """
        if isinstance(self.dtype, PsNumericType):
            return self.dtype.is_int() or self.dtype.is_float() or super().is_real
        else:
            return super().is_real


class BooleanCastFunc(CastFunc, sp.logic.boolalg.Boolean):
    # TODO: documentation
    pass


class VectorMemoryAccess(CastFunc):
    """
    Special memory access for vectorized kernel.
    Arguments: read/write expression, type, aligned, non-temporal, mask (or none), stride
    """

    nargs = (6,)


class ReinterpretCastFunc(CastFunc):
    """
    Reinterpret cast is necessary for the StructType
    """

    pass


class PointerArithmeticFunc(sp.Function, sp.logic.boolalg.Boolean):
    # TODO: documentation, or deprecate!
    @property
    def canonical(self):
        if hasattr(self.args[0], "canonical"):
            return self.args[0].canonical
        else:
            raise NotImplementedError()
