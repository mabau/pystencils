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

from .types import PsAbstractType
from .typed_expressions import ExprOrConstant


class PsFunction(pb.FunctionSymbol, ABC):
    @property
    @abstractmethod
    def arg_count(self) -> int:
        "Number of arguments this function takes"


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
    return Cast(target_type)(ExprOrConstant)
