from typing import Any
from dataclasses import dataclass

from .symbols import PsSymbol


@dataclass
class PsKernelParamsConstraint:
    condition: Any  # FIXME Implement conditions
    message: str = ""

    def to_code(self):
        raise NotImplementedError()

    def get_symbols(self) -> set[PsSymbol]:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"{self.message} [{self.condition}]"
