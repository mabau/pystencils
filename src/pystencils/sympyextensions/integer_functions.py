import sympy as sp
import warnings
from pystencils.sympyextensions import is_integer_sequence


class IntegerFunctionTwoArgsMixIn(sp.Function):
    is_integer = True

    def __new__(cls, arg1, arg2):
        args = [arg1, arg2]
        return super().__new__(cls, *args)

    def _eval_evalf(self, *pargs, **kwargs):
        arg1 = self.args[0].evalf(*pargs, **kwargs) if hasattr(self.args[0], 'evalf') else self.args[0]
        arg2 = self.args[1].evalf(*pargs, **kwargs) if hasattr(self.args[1], 'evalf') else self.args[1]
        return self._eval_op(arg1, arg2)

    def _eval_op(self, arg1, arg2):
        return self


# noinspection PyPep8Naming
class bitwise_xor(IntegerFunctionTwoArgsMixIn):
    pass


# noinspection PyPep8Naming
class bit_shift_right(IntegerFunctionTwoArgsMixIn):
    pass


# noinspection PyPep8Naming
class bit_shift_left(IntegerFunctionTwoArgsMixIn):
    pass


# noinspection PyPep8Naming
class bitwise_and(IntegerFunctionTwoArgsMixIn):
    pass


# noinspection PyPep8Naming
class bitwise_or(IntegerFunctionTwoArgsMixIn):
    pass


# noinspection PyPep8Naming
class int_div(IntegerFunctionTwoArgsMixIn):
    """C-style round-to-zero integer division"""

    def _eval_op(self, arg1, arg2):
        from ..utils import c_intdiv

        return c_intdiv(arg1, arg2)


class int_rem(IntegerFunctionTwoArgsMixIn):
    """C-style round-to-zero integer remainder"""

    def _eval_op(self, arg1, arg2):
        from ..utils import c_rem

        return c_rem(arg1, arg2)


# noinspection PyPep8Naming
# TODO: What do the *two* arguments mean?
#       Apparently, the second is required but ignored?
class int_power_of_2(IntegerFunctionTwoArgsMixIn):
    pass


# noinspection PyPep8Naming
class round_to_multiple_towards_zero(IntegerFunctionTwoArgsMixIn):
    """Returns the next smaller/equal in magnitude integer divisible by given
    divisor.

    Examples:
        >>> round_to_multiple_towards_zero(9, 4)
        8
        >>> round_to_multiple_towards_zero(11, -4)
        8
        >>> round_to_multiple_towards_zero(12, 4)
        12
        >>> round_to_multiple_towards_zero(-9, 4)
        -8
        >>> round_to_multiple_towards_zero(-9, -4)
        -8
    """

    @classmethod
    def eval(cls, arg1, arg2):
        from ..utils import c_intdiv

        if is_integer_sequence((arg1, arg2)):
            return c_intdiv(arg1, arg2) * arg2

    def _eval_op(self, arg1, arg2):
        return self.eval(arg1, arg2)


# noinspection PyPep8Naming
class ceil_to_multiple(IntegerFunctionTwoArgsMixIn):
    """For positive input, returns the next greater/equal integer divisible
    by given divisor. The return value is unspecified if either argument is
    negative.

    Examples:
        >>> ceil_to_multiple(9, 4)
        12
        >>> ceil_to_multiple(11, 4)
        12
        >>> ceil_to_multiple(12, 4)
        12
    """

    @classmethod
    def eval(cls, arg1, arg2):
        from ..utils import c_intdiv

        if is_integer_sequence((arg1, arg2)):
            return c_intdiv(arg1 + arg2 - 1, arg2) * arg2

    def _eval_op(self, arg1, arg2):
        return self.eval(arg1, arg2)


# noinspection PyPep8Naming
class div_ceil(IntegerFunctionTwoArgsMixIn):
    """For positive input, integer division that is always rounded up, i.e.
    `div_ceil(a, b) = ceil(div(a, b))`. The return value is unspecified if
    either argument is negative.

    Examples:
        >>> div_ceil(9, 4)
        3
        >>> div_ceil(8, 4)
        2
    """

    @classmethod
    def eval(cls, arg1, arg2):
        from ..utils import c_intdiv

        if is_integer_sequence((arg1, arg2)):
            return c_intdiv(arg1 + arg2 - 1, arg2)

    def _eval_op(self, arg1, arg2):
        return self.eval(arg1, arg2)


# Deprecated functions.


# noinspection PyPep8Naming
class modulo_floor:
    def __new__(cls, integer, divisor):
        warnings.warn(
            "`modulo_floor` is deprecated. Use `round_to_multiple_towards_zero` instead.",
            DeprecationWarning,
        )
        return round_to_multiple_towards_zero(integer, divisor)


# noinspection PyPep8Naming
class modulo_ceil(sp.Function):
    def __new__(cls, integer, divisor):
        warnings.warn(
            "`modulo_ceil` is deprecated. Use `ceil_to_multiple` instead.",
            DeprecationWarning,
        )
        return ceil_to_multiple(integer, divisor)


# noinspection PyPep8Naming
class div_floor(sp.Function):
    def __new__(cls, integer, divisor):
        warnings.warn(
            "`div_floor` is deprecated. Use `int_div` instead.",
            DeprecationWarning,
        )
        return int_div(integer, divisor)
