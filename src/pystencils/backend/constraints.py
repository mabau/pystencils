from __future__ import annotations

from typing import Any, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .kernelfunction import KernelParameter


@dataclass
class KernelParamsConstraint:
    condition: Any  # FIXME Implement conditions
    message: str = ""

    def to_code(self):
        raise NotImplementedError()

    def get_parameters(self) -> set[KernelParameter]:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"{self.message} [{self.condition}]"
