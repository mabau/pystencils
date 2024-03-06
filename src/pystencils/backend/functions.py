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

from __future__ import annotations
from typing import Any, TYPE_CHECKING
from abc import ABC
from enum import Enum

if TYPE_CHECKING:
    from .ast.expressions import PsExpression


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

    Pow = ("pow", 2)

    def __init__(self, func_name, arg_count):
        self.function_name = func_name
        self.arg_count = arg_count


class PsFunction(ABC):
    __match_args__ = ("name", "arg_count")

    def __init__(self, name: str, num_args: int):
        self._name = name
        self._num_args = num_args

    @property
    def name(self) -> str:
        return self._name

    @property
    def arg_count(self) -> int:
        "Number of arguments this function takes"
        return self._num_args

    def __call__(self, *args: PsExpression) -> Any:
        from .ast.expressions import PsCall

        return PsCall(self, args)


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

    def __init__(self, func: MathFunctions) -> None:
        self._func = func

    @property
    def func(self) -> MathFunctions:
        return self._func

    @property
    def arg_count(self) -> int:
        return self._func.arg_count
