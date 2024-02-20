"""
Functions supported by pystencils.

Every supported function might require handling logic in the following modules:

 - In `freeze.FreezeExpressions`, a case in `map_Function` or a separate mapper method to catch its frontend variant
 - In each backend platform, a case in `Platform.materialize_functions` to map the function onto a concrete
   C/C++ implementation
 - If very special typing rules apply, a case in `typification.Typifier`.

In most cases, typification of function applications will require no special handling.

TODO: Maybe add a way for the user to register additional functions
TODO: Figure out the best way to describe function signatures and overloads for typing
"""

from sys import intern
import pymbolic.primitives as pb
from abc import ABC, abstractmethod
from enum import Enum

from .types import PsAbstractType
from .typed_expressions import ExprOrConstant


class MathFunctions(Enum):
    """Mathematical functions supported by the backend.

    Each platform has to materialize these functions to a concrete implementation.
    """

    Exp = ("exp", 1)
    Sin = ("sin", 1)
    Cos = ("cos", 1)
    Tan = ("tan", 1)

    Abs = ("abs", 1)

    Min = ("min", 2)
    Max = ("max", 2)

    def __init__(self, func_name, arg_count):
        self.function_name = func_name
        self.arg_count = arg_count


class PsFunction(pb.FunctionSymbol, ABC):
    @property
    @abstractmethod
    def arg_count(self) -> int:
        "Number of arguments this function takes"


class CFunction(PsFunction):
    """A concrete C function."""

    def __init__(self, qualified_name: str, arg_count: int):
        self._qname = qualified_name
        self._arg_count = arg_count

    @property
    def qualified_name(self) -> str:
        return self._qname

    @property
    def arg_count(self) -> int:
        return self._arg_count


class PsMathFunction(PsFunction):
    """Homogenously typed mathematical functions."""

    init_arg_names = ("func",)
    mapper_method = intern("map_math_function")

    def __init__(self, func: MathFunctions) -> None:
        self._func = func

    @property
    def func(self) -> MathFunctions:
        return self._func

    @property
    def arg_count(self) -> int:
        return self._func.arg_count


class Deref(PsFunction):
    """Dereferences a pointer."""

    mapper_method = intern("map_deref")

    @property
    def arg_count(self) -> int:
        return 1


deref = Deref()


class AddressOf(PsFunction):
    """Take the address of an object"""

    mapper_method = intern("map_address_of")

    @property
    def arg_count(self) -> int:
        return 1


address_of = AddressOf()


class Cast(PsFunction):
    mapper_method = intern("map_cast")

    """An unsafe C-style type cast"""

    def __init__(self, target_type: PsAbstractType):
        self._target_type = target_type

    @property
    def arg_count(self) -> int:
        return 1

    @property
    def target_type(self) -> PsAbstractType:
        return self._target_type


def cast(target_type: PsAbstractType, arg: ExprOrConstant):
    return Cast(target_type)(arg)
