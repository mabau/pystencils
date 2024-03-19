import sympy as sp

from ..types import PsType, PsNumericType, PsPointerType, PsBoolType
from ..types.quick import create_type


def assumptions_from_dtype(dtype: PsType):
    """Derives SymPy assumptions from :class:`PsAbstractType`

    Args:
        dtype (PsAbstractType): a pystencils data type
    Returns:
        A dict of SymPy assumptions
    """
    assumptions = dict()

    if isinstance(dtype, PsNumericType):
        if dtype.is_int():
            assumptions.update({"integer": True})
        if dtype.is_uint():
            assumptions.update({"negative": False})
        if dtype.is_int() or dtype.is_float():
            assumptions.update({"real": True})

    return assumptions


def is_loop_counter_symbol(symbol):
    from ..defaults import DEFAULTS

    try:
        return DEFAULTS.spatial_counters.index(symbol)
    except ValueError:
        return None


class PsTypeAtom(sp.Atom):
    """Wrapper around a PsType to disguise it as a SymPy atom."""

    def __new__(cls, *args, **kwargs):
        return sp.Basic.__new__(cls)
    
    def __init__(self, dtype: PsType) -> None:
        self._dtype = dtype

    def _sympystr(self, *args, **kwargs):
        return str(self._dtype)

    def get(self) -> PsType:
        return self._dtype


class TypedSymbol(sp.Symbol):
    def __new__(cls, *args, **kwds):
        obj = TypedSymbol.__xnew_cached_(cls, *args, **kwds)
        return obj

    def __new_stage2__(
        cls, name, dtype, **kwargs
    ):  # TODO does not match signature of sp.Symbol???
        # TODO: also Symbol should be allowed  ---> see sympy Variable
        dtype = create_type(dtype)
        assumptions = assumptions_from_dtype(dtype)
        assumptions.update(kwargs)

        obj = super(TypedSymbol, cls).__xnew__(cls, name, **assumptions)
        obj._dtype = create_type(dtype)

        return obj

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(sp.core.cacheit(__new_stage2__))

    @property
    def dtype(self) -> PsType:
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
        return self.dtype.required_headers


class FieldStrideSymbol(TypedSymbol):
    """Sympy symbol representing the stride value of a field in a specific coordinate."""

    def __new__(cls, *args, **kwds):
        obj = FieldStrideSymbol.__xnew_cached_(cls, *args, **kwds)
        return obj

    def __new_stage2__(cls, field_name: str, coordinate: int):
        from ..defaults import DEFAULTS

        name = f"_stride_{field_name}_{coordinate}"
        obj = super(FieldStrideSymbol, cls).__xnew__(
            cls, name, DEFAULTS.index_dtype, positive=True
        )
        obj.field_name = field_name
        obj.coordinate = coordinate
        return obj

    def __getnewargs__(self):
        return self.field_name, self.coordinate

    def __getnewargs_ex__(self):
        return (self.field_name, self.coordinate), {}

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(sp.core.cacheit(__new_stage2__))

    def _hashable_content(self):
        return super()._hashable_content(), self.coordinate, self.field_name


class FieldShapeSymbol(TypedSymbol):
    """Sympy symbol representing the shape value of a sequence of fields. In a kernel iterating over multiple fields
    there is only one set of `FieldShapeSymbol`s since all the fields have to be of equal size.
    """

    def __new__(cls, *args, **kwds):
        obj = FieldShapeSymbol.__xnew_cached_(cls, *args, **kwds)
        return obj

    def __new_stage2__(cls, field_name: str, coordinate: int):
        from ..defaults import DEFAULTS

        name = f"_size_{field_name}_{coordinate}"
        obj = super(FieldShapeSymbol, cls).__xnew__(
            cls, name, DEFAULTS.index_dtype, positive=True
        )
        obj.field_name = field_name
        obj.coordinate = coordinate
        return obj

    def __getnewargs__(self):
        return self.field_name, self.coordinate

    def __getnewargs_ex__(self):
        return (self.field_name, self.coordinate), {}

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(sp.core.cacheit(__new_stage2__))

    def _hashable_content(self):
        return super()._hashable_content(), self.coordinate, self.field_name


class FieldPointerSymbol(TypedSymbol):
    """Sympy symbol representing the pointer to the beginning of the field data."""

    def __new__(cls, *args, **kwds):
        obj = FieldPointerSymbol.__xnew_cached_(cls, *args, **kwds)
        return obj

    def __new_stage2__(cls, field_name, field_dtype: PsType, const: bool):
        name = f"_data_{field_name}"
        dtype = PsPointerType(field_dtype, const=const, restrict=True)
        obj = super(FieldPointerSymbol, cls).__xnew__(cls, name, dtype)
        obj.field_name = field_name
        return obj

    def __getnewargs__(self):
        return self.field_name, self.dtype, self.dtype.const

    def __getnewargs_ex__(self):
        return (self.field_name, self.dtype, self.dtype.const), {}

    def _hashable_content(self):
        return super()._hashable_content(), self.field_name

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(sp.core.cacheit(__new_stage2__))


class CastFunc(sp.Function):
    """
    CastFunc is used in order to introduce static casts. They are especially useful as a way to signal what type
    a certain node should have, if it is impossible to add a type to a node, e.g. a sp.Number.
    """

    is_Atom = True

    def __new__(cls, *args, **kwargs):
        if len(args) != 2:
            pass
        expr, dtype, *other_args = args

        # If we have two consecutive casts, throw the inner one away.
        # This optimisation is only available for simple casts. Thus the == is intended here!
        if expr.__class__ == CastFunc:
            expr = expr.args[0]

        if not isinstance(dtype, PsTypeAtom):
            dtype = PsTypeAtom(create_type(dtype))
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
    def dtype(self) -> PsType:
        assert isinstance(self.args[1], PsTypeAtom)
        return self.args[1].get()

    @property
    def expr(self):
        return self.args[0]

    @property
    def is_integer(self):
        if isinstance(self.dtype, PsNumericType):
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
