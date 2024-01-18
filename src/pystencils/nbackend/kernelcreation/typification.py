import pymbolic.primitives as pb
from pymbolic.mapper import Mapper

from .context import KernelCreationContext
from ..types import PsAbstractType
from ..typed_expressions import PsTypedVariable


class Typifier(Mapper):
    def __init__(self, ctx: KernelCreationContext):
        self._ctx = ctx

    def __call__(self, expr: pb.Expression) -> tuple[pb.Expression, PsAbstractType]:
        return self.rec(expr)

    def map_variable(self, var: pb.Variable) -> tuple[pb.Expression, PsAbstractType]:
        dtype = NotImplemented  # determine variable type
        return PsTypedVariable(var.name, dtype), dtype
