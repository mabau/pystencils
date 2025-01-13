from __future__ import annotations
from typing import TYPE_CHECKING

from pystencils.backend.ast.astnode import PsAstNode
from pystencils.backend.constants import PsConstant
from pystencils.backend.emission.base_printer import PrinterCtx, EmissionError
from pystencils.backend.memory import PsSymbol
from .base_printer import BasePrinter

from ...types import PsType, PsArrayType, PsScalarType, PsTypeError
from ..ast.expressions import PsBufferAcc
from ..ast.vector import PsVecMemAcc

if TYPE_CHECKING:
    from ...codegen import Kernel


def emit_code(ast: PsAstNode | Kernel):
    printer = CAstPrinter()
    return printer(ast)


class CAstPrinter(BasePrinter):

    def __init__(self, indent_width=3):
        super().__init__(indent_width)

    def visit(self, node: PsAstNode, pc: PrinterCtx) -> str:
        match node:
            case PsVecMemAcc():
                raise EmissionError(
                    f"Unable to print C code for vector memory access {node}.\n"
                    f"Vectorized memory accesses must be mapped to intrinsics before emission."
                )

            case PsBufferAcc():
                raise EmissionError(
                    f"Unable to print C code for buffer access {node}.\n"
                    f"Buffer accesses must be lowered using the `LowerToC` pass before emission."
                )

            case _:
                return super().visit(node, pc)

    def _symbol_decl(self, symb: PsSymbol):
        dtype = symb.get_dtype()

        if isinstance(dtype, PsArrayType):
            array_dims = dtype.shape
            dtype = dtype.base_type
        else:
            array_dims = ()

        code = f"{self._type_str(dtype)} {symb.name}"
        for d in array_dims:
            code += f"[{str(d) if d is not None else ''}]"

        return code

    def _constant_literal(self, constant: PsConstant):
        dtype = constant.get_dtype()
        if not isinstance(dtype, PsScalarType):
            raise EmissionError("Cannot print literals for non-scalar constants.")

        return dtype.create_literal(constant.value)

    def _type_str(self, dtype: PsType):
        try:
            return dtype.c_string()
        except PsTypeError:
            raise EmissionError(f"Unable to print type {dtype} as a C data type.")
