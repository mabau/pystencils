import numpy as np
import sympy as sp
from sympy.logic.boolalg import Boolean

from pystencils.typing.types import AbstractType, BasicType
from pystencils.typing.typed_sympy import TypedSymbol


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
        if not isinstance(dtype, AbstractType):
            dtype = BasicType(dtype)
        # to work in conditions of sp.Piecewise cast_func has to be of type Boolean as well
        # however, a cast_function should only be a boolean if its argument is a boolean, otherwise this leads
        # to problems when for example comparing cast_func's for equality
        #
        # lhs = bitwise_and(a, cast_func(1, 'int'))
        # rhs = cast_func(0, 'int')
        # print( sp.Ne(lhs, rhs) ) # would give true if all cast_funcs are booleans
        # -> thus a separate class boolean_cast_func is introduced
        if isinstance(expr, Boolean) and (not isinstance(expr, TypedSymbol) or expr.dtype == BasicType('bool')):
            cls = BooleanCastFunc

        return sp.Function.__new__(cls, expr, dtype, *other_args, **kwargs)

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
        return self.args[1]

    @property
    def expr(self):
        return self.args[0]

    @property
    def is_integer(self):
        """
        Uses Numpy type hierarchy to determine :func:`sympy.Expr.is_integer` predicate

        For reference: Numpy type hierarchy https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.scalars.html
        """
        if hasattr(self.dtype, 'numpy_dtype'):
            return np.issubdtype(self.dtype.numpy_dtype, np.integer) or super().is_integer
        else:
            return super().is_integer

    @property
    def is_negative(self):
        """
        See :func:`.TypedSymbol.is_integer`
        """
        if hasattr(self.dtype, 'numpy_dtype'):
            if np.issubdtype(self.dtype.numpy_dtype, np.unsignedinteger):
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
        if hasattr(self.dtype, 'numpy_dtype'):
            return np.issubdtype(self.dtype.numpy_dtype, np.integer) or np.issubdtype(self.dtype.numpy_dtype,
                                                                                      np.floating) or super().is_real
        else:
            return super().is_real


class BooleanCastFunc(CastFunc, Boolean):
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


class PointerArithmeticFunc(sp.Function, Boolean):
    # TODO: documentation, or deprecate!
    @property
    def canonical(self):
        if hasattr(self.args[0], 'canonical'):
            return self.args[0].canonical
        else:
            raise NotImplementedError()
