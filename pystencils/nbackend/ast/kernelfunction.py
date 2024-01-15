from typing import Sequence

from typing import Generator
from .nodes import PsAstNode, PsBlock, failing_cast
from ..typed_expressions import PsTypedVariable
from ...enums import Target


class PsKernelFunction(PsAstNode):
    """A complete pystencils kernel function."""

    __match_args__ = ("body",)

    def __init__(self, body: PsBlock, target: Target, name: str = "kernel"):
        self._body = body
        self._target = target
        self._name = name

    @property
    def target(self) -> Target:
        """See pystencils.Target"""
        return self._target

    @property
    def body(self) -> PsBlock:
        return self._body

    @body.setter
    def body(self, body: PsBlock):
        self._body = body

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    def num_children(self) -> int:
        return 1

    def children(self) -> Generator[PsAstNode, None, None]:
        yield from (self._body,)

    def get_child(self, idx: int):
        if idx not in (0, -1):
            raise IndexError(f"Child index out of bounds: {idx}")
        return self._body

    def set_child(self, idx: int, c: PsAstNode):
        if idx not in (0, -1):
            raise IndexError(f"Child index out of bounds: {idx}")
        self._body = failing_cast(PsBlock, c)

    def get_parameters(self) -> Sequence[PsTypedVariable]:
        """Collect the list of parameters to this function.

        This function performs a full traversal of the AST.
        To improve performance, make sure to cache the result if necessary.
        """
        from .analysis import UndefinedVariablesCollector

        params = UndefinedVariablesCollector().collect(self)
        return sorted(params, key=lambda p: p.name)
