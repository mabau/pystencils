"""
Functions supported by pystencils.

Every supported function might require handling logic in the following modules:

 - In `freeze.FreezeExpressions`, a case in `map_Function` or a separate mapper method to catch its frontend variant
 - In each backend platform, a case in `materialize_functions` to map the function onto a concrete C/C++ implementation
 - If very special typing rules apply, a case in `typification.Typifier`.

In most cases, typification of function applications will require no special handling.

TODO: Maybe add a way for the user to register additional functions
TODO: Figure out the best way to describe function signatures and overloads for typing
"""

import pymbolic.primitives as pb
from abc import ABC, abstractmethod


class PsFunction(pb.FunctionSymbol, ABC):
    @property
    @abstractmethod
    def arg_count(self) -> int:
        "Number of arguments this function takes"
