import sympy as sp
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
class modulo_floor(sp.Function):
    """Returns the next smaller integer divisible by given divisor.

    Examples:
        >>> modulo_floor(9, 4)
        8
        >>> modulo_floor(11, 4)
        8
        >>> modulo_floor(12, 4)
        12
    """
    nargs = 2
    is_integer = True

    def __new__(cls, integer, divisor):
        if is_integer_sequence((integer, divisor)):
            return (int(integer) // int(divisor)) * divisor
        else:
            return super().__new__(cls, integer, divisor)

    #   TODO: Implement this in FreezeExpressions
    # def to_c(self, print_func):
    #     dtype = collate_types((get_type_of_expression(self.args[0]), get_type_of_expression(self.args[1])))
    #     assert dtype.is_int()
    #     return "({dtype})(({0}) / ({1})) * ({1})".format(print_func(self.args[0]),
    #                                                      print_func(self.args[1]), dtype=dtype)


# noinspection PyPep8Naming
class modulo_ceil(sp.Function):
    """Returns the next bigger integer divisible by given divisor.

    Examples:
        >>> modulo_ceil(9, 4)
        12
        >>> modulo_ceil(11, 4)
        12
        >>> modulo_ceil(12, 4)
        12
    """
    nargs = 2
    is_integer = True

    def __new__(cls, integer, divisor):
        if is_integer_sequence((integer, divisor)):
            return integer if integer % divisor == 0 else ((integer // divisor) + 1) * divisor
        else:
            return super().__new__(cls, integer, divisor)

    #   TODO: Implement this in FreezeExpressions
    # def to_c(self, print_func):
    #     dtype = collate_types((get_type_of_expression(self.args[0]), get_type_of_expression(self.args[1])))
    #     assert dtype.is_int()
    #     code = "(({0}) % ({1}) == 0 ? {0} : (({dtype})(({0}) / ({1}))+1) * ({1}))"
    #     return code.format(print_func(self.args[0]), print_func(self.args[1]), dtype=dtype)


# noinspection PyPep8Naming
class div_ceil(sp.Function):
    """Integer division that is always rounded up

    Examples:
        >>> div_ceil(9, 4)
        3
        >>> div_ceil(8, 4)
        2
    """
    nargs = 2
    is_integer = True

    def __new__(cls, integer, divisor):
        if is_integer_sequence((integer, divisor)):
            return integer // divisor if integer % divisor == 0 else (integer // divisor) + 1
        else:
            return super().__new__(cls, integer, divisor)

    #   TODO: Implement this in FreezeExpressions
    # def to_c(self, print_func):
    #     dtype = collate_types((get_type_of_expression(self.args[0]), get_type_of_expression(self.args[1])))
    #     assert dtype.is_int()
    #     code = "( ({0}) % ({1}) == 0 ? ({dtype})({0}) / ({dtype})({1}) : ( ({dtype})({0}) / ({dtype})({1}) ) +1 )"
    #     return code.format(print_func(self.args[0]), print_func(self.args[1]), dtype=dtype)


# noinspection PyPep8Naming
class div_floor(sp.Function):
    """Integer division

    Examples:
        >>> div_floor(9, 4)
        2
        >>> div_floor(8, 4)
        2
    """
    nargs = 2
    is_integer = True

    def __new__(cls, integer, divisor):
        if is_integer_sequence((integer, divisor)):
            return integer // divisor
        else:
            return super().__new__(cls, integer, divisor)

    #   TODO: Implement this in FreezeExpressions
    # def to_c(self, print_func):
    #     dtype = collate_types((get_type_of_expression(self.args[0]), get_type_of_expression(self.args[1])))
    #     assert dtype.is_int()
    #     code = "(({dtype})({0}) / ({dtype})({1}))"
    #     return code.format(print_func(self.args[0]), print_func(self.args[1]), dtype=dtype)
