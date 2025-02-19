from __future__ import annotations
from typing import Sequence, Any

from .parameters import Parameter
from ..types import PsType

from ..backend.ast.expressions import PsExpression


class Lambda:
    """A one-line function emitted by the code generator as an auxiliary object."""

    def __init__(self, expr: PsExpression, params: Sequence[Parameter]):
        self._expr = expr
        self._params = tuple(params)
        self._return_type = expr.get_dtype()

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        """Parameters to this lambda"""
        return self._params

    @property
    def return_type(self) -> PsType:
        """Return type of this lambda"""
        return self._return_type

    def __call__(self, **kwargs) -> Any:
        """Evaluate this lambda with the given arguments.

        The lambda must receive a value for each parameter listed in `parameters`.
        """
        from ..backend.ast.expressions import evaluate_expression

        return evaluate_expression(self._expr, kwargs)

    def __str__(self) -> str:
        return str(self._expr)

    def c_code(self) -> str:
        """Print the C code of this lambda"""
        from ..backend.emission import CAstPrinter

        printer = CAstPrinter()
        return printer(self._expr)
