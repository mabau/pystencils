from __future__ import annotations

from abc import ABC
from typing import Callable, Sequence

from .ast.structural import PsBlock

from .constraints import KernelParamsConstraint
from ..types import PsType
from .jit import JitBase, no_jit

from ..enums import Target
from ..field import Field


class KernelParameter:
    __match_args__ = ("name", "dtype")

    def __init__(self, name: str, dtype: PsType):
        self._name = name
        self._dtype = dtype

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    def _hashable_contents(self):
        return (self._name, self._dtype)

    def __hash__(self) -> int:
        return hash(self._hashable_contents())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KernelParameter):
            return False

        return (
            type(self) is type(other)
            and self._hashable_contents() == other._hashable_contents()
        )

    def __str__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name = {self._name}, dtype = {self._dtype})"


class FieldParameter(KernelParameter, ABC):
    __match_args__ = KernelParameter.__match_args__ + ("field",)

    def __init__(self, name: str, dtype: PsType, field: Field):
        super().__init__(name, dtype)
        self._field = field

    @property
    def field(self):
        return self._field

    def _hashable_contents(self):
        return super()._hashable_contents() + (self._field,)


class FieldShapeParam(FieldParameter):
    __match_args__ = FieldParameter.__match_args__ + ("coordinate",)

    def __init__(self, name: str, dtype: PsType, field: Field, coordinate: int):
        super().__init__(name, dtype, field)
        self._coordinate = coordinate

    @property
    def coordinate(self):
        return self._coordinate

    def _hashable_contents(self):
        return super()._hashable_contents() + (self._coordinate,)


class FieldStrideParam(FieldParameter):
    __match_args__ = FieldParameter.__match_args__ + ("coordinate",)

    def __init__(self, name: str, dtype: PsType, field: Field, coordinate: int):
        super().__init__(name, dtype, field)
        self._coordinate = coordinate

    @property
    def coordinate(self):
        return self._coordinate

    def _hashable_contents(self):
        return super()._hashable_contents() + (self._coordinate,)


class FieldPointerParam(FieldParameter):
    def __init__(self, name: str, dtype: PsType, field: Field):
        super().__init__(name, dtype, field)


class KernelFunction:
    """A pystencils kernel function.

    The kernel function is the final result of the translation process.
    It is immutable, and its AST should not be altered any more, either, as this
    might invalidate information about the kernel already stored in the `KernelFunction` object.
    """

    def __init__(
        self,
        body: PsBlock,
        target: Target,
        name: str,
        parameters: Sequence[KernelParameter],
        required_headers: set[str],
        constraints: Sequence[KernelParamsConstraint],
        jit: JitBase = no_jit,
    ):
        self._body: PsBlock = body
        self._target = target
        self._name = name
        self._params = tuple(parameters)
        self._required_headers = required_headers
        self._constraints = tuple(constraints)
        self._jit = jit

    @property
    def body(self) -> PsBlock:
        return self._body

    @property
    def target(self) -> Target:
        return self._target

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, n: str):
        self._name = n

    @property
    def function_name(self) -> str:
        """For backward compatibility"""
        return self._name

    @property
    def parameters(self) -> tuple[KernelParameter, ...]:
        return self._params

    @property
    def required_headers(self) -> set[str]:
        return self._required_headers

    @property
    def constraints(self) -> tuple[KernelParamsConstraint, ...]:
        return self._constraints

    def compile(self) -> Callable[..., None]:
        return self._jit.compile(self)
