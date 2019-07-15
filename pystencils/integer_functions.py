import sympy as sp

from pystencils.data_types import collate_types, get_type_of_expression
from pystencils.sympyextensions import is_integer_sequence

bitwise_xor = sp.Function("bitwise_xor")
bit_shift_right = sp.Function("bit_shift_right")
bit_shift_left = sp.Function("bit_shift_left")
bitwise_and = sp.Function("bitwise_and")
bitwise_or = sp.Function("bitwise_or")
int_div = sp.Function("int_div")
int_power_of_2 = sp.Function("int_power_of_2")


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
        >>> from pystencils import TypedSymbol
        >>> a, b = TypedSymbol("a", "int64"), TypedSymbol("b", "int32")
        >>> modulo_floor(a, b).to_c(str)
        '(int64_t)((a) / (b)) * (b)'
    """
    nargs = 2

    def __new__(cls, integer, divisor):
        if is_integer_sequence((integer, divisor)):
            return (int(integer) // int(divisor)) * divisor
        else:
            return super().__new__(cls, integer, divisor)

    def to_c(self, print_func):
        dtype = collate_types((get_type_of_expression(self.args[0]), get_type_of_expression(self.args[1])))
        assert dtype.is_int()
        return "({dtype})(({0}) / ({1})) * ({1})".format(print_func(self.args[0]),
                                                         print_func(self.args[1]), dtype=dtype)


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
        >>> from pystencils import TypedSymbol
        >>> a, b = TypedSymbol("a", "int64"), TypedSymbol("b", "int32")
        >>> modulo_ceil(a, b).to_c(str)
        '((a) % (b) == 0 ? a : ((int64_t)((a) / (b))+1) * (b))'
    """
    nargs = 2

    def __new__(cls, integer, divisor):
        if is_integer_sequence((integer, divisor)):
            return integer if integer % divisor == 0 else ((integer // divisor) + 1) * divisor
        else:
            return super().__new__(cls, integer, divisor)

    def to_c(self, print_func):
        dtype = collate_types((get_type_of_expression(self.args[0]), get_type_of_expression(self.args[1])))
        assert dtype.is_int()
        code = "(({0}) % ({1}) == 0 ? {0} : (({dtype})(({0}) / ({1}))+1) * ({1}))"
        return code.format(print_func(self.args[0]), print_func(self.args[1]), dtype=dtype)


# noinspection PyPep8Naming
class div_ceil(sp.Function):
    """Integer division that is always rounded up

    Examples:
        >>> div_ceil(9, 4)
        3
        >>> div_ceil(8, 4)
        2
        >>> from pystencils import TypedSymbol
        >>> a, b = TypedSymbol("a", "int64"), TypedSymbol("b", "int32")
        >>> div_ceil(a, b).to_c(str)
        '( (a) % (b) == 0 ? (int64_t)(a) / (int64_t)(b) : ( (int64_t)(a) / (int64_t)(b) ) +1 )'
    """
    nargs = 2

    def __new__(cls, integer, divisor):
        if is_integer_sequence((integer, divisor)):
            return integer // divisor if integer % divisor == 0 else (integer // divisor) + 1
        else:
            return super().__new__(cls, integer, divisor)

    def to_c(self, print_func):
        dtype = collate_types((get_type_of_expression(self.args[0]), get_type_of_expression(self.args[1])))
        assert dtype.is_int()
        code = "( ({0}) % ({1}) == 0 ? ({dtype})({0}) / ({dtype})({1}) : ( ({dtype})({0}) / ({dtype})({1}) ) +1 )"
        return code.format(print_func(self.args[0]), print_func(self.args[1]), dtype=dtype)


# noinspection PyPep8Naming
class div_floor(sp.Function):
    """Integer division

    Examples:
        >>> div_floor(9, 4)
        2
        >>> div_floor(8, 4)
        2
        >>> from pystencils import TypedSymbol
        >>> a, b = TypedSymbol("a", "int64"), TypedSymbol("b", "int32")
        >>> div_floor(a, b).to_c(str)
        '((int64_t)(a) / (int64_t)(b))'
    """
    nargs = 2

    def __new__(cls, integer, divisor):
        if is_integer_sequence((integer, divisor)):
            return integer // divisor
        else:
            return super().__new__(cls, integer, divisor)

    def to_c(self, print_func):
        dtype = collate_types((get_type_of_expression(self.args[0]), get_type_of_expression(self.args[1])))
        assert dtype.is_int()
        code = "(({dtype})({0}) / ({dtype})({1}))"
        return code.format(print_func(self.args[0]), print_func(self.args[1]), dtype=dtype)
