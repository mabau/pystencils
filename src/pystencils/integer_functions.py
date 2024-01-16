# TODO #47 move to a module functions
import numpy as np
import sympy as sp

from pystencils.typing import CastFunc, collate_types, create_type, get_type_of_expression
from pystencils.sympyextensions import is_integer_sequence


class IntegerFunctionTwoArgsMixIn(sp.Function):
    is_integer = True

    def __new__(cls, arg1, arg2):
        args = []
        for a in (arg1, arg2):
            if isinstance(a, sp.Number) or isinstance(a, int):
                args.append(CastFunc(a, create_type("int")))
            elif isinstance(a, np.generic):
                args.append(CastFunc(a, a.dtype))
            else:
                args.append(a)

        for a in args:
            try:
                dtype = get_type_of_expression(a)
                if not dtype.is_int():
                    raise ValueError("Argument to integer function is not an int but " + str(dtype))
            except NotImplementedError:
                raise ValueError("Integer functions can only be constructed with typed expressions")
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
    
    def _eval_op(self, arg1, arg2):
        return int(arg1 // arg2)


# noinspection PyPep8Naming
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
        >>> from pystencils import TypedSymbol
        >>> a, b = TypedSymbol("a", "int64"), TypedSymbol("b", "int32")
        >>> modulo_floor(a, b).to_c(str)
        '(int64_t)((a) / (b)) * (b)'
    """
    nargs = 2
    is_integer = True

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
    is_integer = True

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
    is_integer = True

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
    is_integer = True

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
