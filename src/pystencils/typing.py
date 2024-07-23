from .sympyextensions import TypedSymbol as _TypedSymbol
from .types import create_type as _create_type

from warnings import warn
warn(
    "Importing `TypedSymbol` and `create_type` from `pystencils.typing` is deprecated. "
    "Import from `pystencils` instead."
)

TypedSymbol = _TypedSymbol
create_type = _create_type
