from __future__ import annotations
from typing import TYPE_CHECKING

from pystencils.backend.constants import PsConstant
from pystencils.backend.emission.base_printer import PrinterCtx
from pystencils.backend.memory import PsSymbol
from pystencils.types.meta import PsType, deconstify

from .base_printer import BasePrinter, Ops, LR

from ..ast import PsAstNode
from ..ast.expressions import PsBufferAcc
from ..ast.vector import PsVecMemAcc, PsVecBroadcast

if TYPE_CHECKING:
    from ...codegen import Kernel


def emit_ir(ir: PsAstNode | Kernel):
    """Emit the IR as C-like pseudo-code for inspection."""
    ir_printer = IRAstPrinter()
    return ir_printer(ir)


class IRAstPrinter(BasePrinter):
    """Print the IR AST as pseudo-code.
    
    This printer produces a complete pseudocode representation of a pystencils AST.
    Other than the `CAstPrinter`, the `IRAstPrinter` is capable of emitting code for
    each node defined in `ast <pystencils.backend.ast>`.
    It is furthermore configurable w.r.t. the level of detail it should emit.

    Args:
        indent_width: Number of spaces with which to indent lines in each nested block.
        annotate_constants: If ``True`` (the default), annotate all constant literals with their data type.
    """

    def __init__(self, indent_width=3, annotate_constants: bool = True):
        super().__init__(indent_width)
        self._annotate_constants = annotate_constants

    def visit(self, node: PsAstNode, pc: PrinterCtx) -> str:
        match node:
            case PsBufferAcc(ptr, indices):
                pc.push_op(Ops.Subscript, LR.Left)
                base_code = self.visit(ptr, pc)
                pc.pop_op()

                pc.push_op(Ops.Weakest, LR.Middle)
                indices_code = ", ".join(self.visit(idx, pc) for idx in indices)
                pc.pop_op()

                return pc.parenthesize(
                    base_code + "[" + indices_code + "]", Ops.Subscript
                )

            case PsVecMemAcc(ptr, offset, lanes, stride):
                pc.push_op(Ops.Subscript, LR.Left)
                ptr_code = self.visit(ptr, pc)
                pc.pop_op()

                pc.push_op(Ops.Weakest, LR.Middle)
                offset_code = self.visit(offset, pc)
                pc.pop_op()

                stride_code = "" if stride is None else f", stride={stride}"

                code = f"vec_memacc< {lanes}{stride_code} >({ptr_code}, {offset_code})"
                return pc.parenthesize(code, Ops.Subscript)

            case PsVecBroadcast(lanes, operand):
                pc.push_op(Ops.Weakest, LR.Middle)
                operand_code = self.visit(operand, pc)
                pc.pop_op()

                return pc.parenthesize(
                    f"vec_broadcast<{lanes}>({operand_code})", Ops.Weakest
                )

            case _:
                return super().visit(node, pc)

    def _symbol_decl(self, symb: PsSymbol):
        return f"{symb.name}: {self._type_str(symb.dtype)}"

    def _constant_literal(self, constant: PsConstant) -> str:
        if self._annotate_constants:
            return f"[{constant.value}: {self._deconst_type_str(constant.dtype)}]"
        else:
            return str(constant.value)

    def _type_str(self, dtype: PsType | None):
        if dtype is None:
            return "<untyped>"
        else:
            return str(dtype)

    def _deconst_type_str(self, dtype: PsType | None):
        if dtype is None:
            return "<untyped>"
        else:
            return str(deconstify(dtype))
