from __future__ import annotations
from typing import Sequence
from enum import Enum
from functools import cache

from ..ast.expressions import (
    PsExpression,
    PsAddressOf,
    PsMemAcc,
    PsUnOp,
    PsBinOp,
    PsAdd,
    PsSub,
    PsMul,
    PsDiv,
    PsConstantExpr
)
from ..ast.vector import PsVecMemAcc, PsVecBroadcast
from ...types import PsCustomType, PsVectorType, PsPointerType
from ..constants import PsConstant

from ..exceptions import MaterializationError
from .generic_cpu import GenericVectorCpu
from ..kernelcreation import KernelCreationContext

from ...types.quick import Fp, SInt
from ..functions import CFunction


class X86VectorArch(Enum):
    SSE = 128
    AVX = 256
    AVX512 = 512
    AVX512_FP16 = AVX512 + 1    # TODO improve modelling?

    def __ge__(self, other: X86VectorArch) -> bool:
        return self.value >= other.value

    def __gt__(self, other: X86VectorArch) -> bool:
        return self.value > other.value

    def __str__(self) -> str:
        return self.name

    @property
    def max_vector_width(self) -> int:
        return self.value

    def intrin_prefix(self, vtype: PsVectorType) -> str:
        match vtype.width:
            case 128 if self >= X86VectorArch.SSE:
                prefix = "_mm"
            case 256 if self >= X86VectorArch.AVX:
                prefix = "_mm256"
            case 512 if self >= X86VectorArch.AVX512:
                prefix = "_mm512"
            case other:
                raise MaterializationError(
                    f"x86/{self} does not support vector width {other}"
                )

        return prefix

    def intrin_suffix(self, vtype: PsVectorType) -> str:
        scalar_type = vtype.scalar_type
        match scalar_type:
            case Fp(16) if self >= X86VectorArch.AVX512_FP16:
                suffix = "ph"
            case Fp(32):
                suffix = "ps"
            case Fp(64):
                suffix = "pd"
            case SInt(width):
                suffix = f"epi{width}"
            case _:
                raise MaterializationError(
                    f"x86/{self} does not support scalar type {scalar_type}"
                )

        return suffix
    
    def intrin_type(self, vtype: PsVectorType):
        scalar_type = vtype.scalar_type
        match scalar_type:
            case Fp(16) if self >= X86VectorArch.AVX512:
                suffix = "h"
            case Fp(32):
                suffix = ""
            case Fp(64):
                suffix = "d"
            case SInt(_):
                suffix = "i"
            case _:
                raise MaterializationError(
                    f"x86/{self} does not support scalar type {scalar_type}"
                )

        if vtype.width > self.max_vector_width:
            raise MaterializationError(
                f"x86/{self} does not support {vtype}"
            )
        return PsCustomType(f"__m{vtype.width}{suffix}")


class X86VectorCpu(GenericVectorCpu):
    """Platform modelling the X86 SSE/AVX/AVX512 vector architectures.

    All intrinsics information is extracted from
    https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html.
    """

    def __init__(self, ctx: KernelCreationContext, vector_arch: X86VectorArch):
        super().__init__(ctx)
        self._vector_arch = vector_arch

    @property
    def vector_arch(self) -> X86VectorArch:
        return self._vector_arch

    @property
    def required_headers(self) -> set[str]:
        if self._vector_arch == X86VectorArch.SSE:
            headers = {
                "<immintrin.h>",
                "<xmmintrin.h>",
                "<emmintrin.h>",
                "<pmmintrin.h>",
                "<tmmintrin.h>",
                "<smmintrin.h>",
                "<nmmintrin.h>",
            }
        else:
            headers = {"<immintrin.h>"}

        return super().required_headers | headers

    def type_intrinsic(self, vector_type: PsVectorType) -> PsCustomType:
        return self._vector_arch.intrin_type(vector_type)

    def constant_intrinsic(self, c: PsConstant) -> PsExpression:
        vtype = c.dtype
        assert isinstance(vtype, PsVectorType)
        stype = vtype.scalar_type

        prefix = self._vector_arch.intrin_prefix(vtype)
        suffix = self._vector_arch.intrin_suffix(vtype)

        if stype == SInt(64) and vtype.vector_entries <= 4:
            suffix += "x"

        set_func = CFunction(
            f"{prefix}_set_{suffix}", (stype,) * vtype.vector_entries, vtype
        )

        values = [PsConstantExpr(PsConstant(v, stype)) for v in c.value]
        return set_func(*values)

    def op_intrinsic(
        self, expr: PsExpression, operands: Sequence[PsExpression]
    ) -> PsExpression:
        match expr:
            case PsUnOp() | PsBinOp():
                func = _x86_op_intrin(self._vector_arch, expr, expr.get_dtype())
                return func(*operands)
            case _:
                raise MaterializationError(f"Cannot map {type(expr)} to x86 intrinsic")

    def vector_load(self, acc: PsVecMemAcc) -> PsExpression:
        if acc.stride is None:
            load_func = _x86_packed_load(self._vector_arch, acc.dtype, False)
            return load_func(
                PsAddressOf(PsMemAcc(acc.pointer, acc.offset))
            )
        else:
            raise NotImplementedError("Gather loads not implemented yet.")

    def vector_store(self, acc: PsVecMemAcc, arg: PsExpression) -> PsExpression:
        if acc.stride is None:
            store_func = _x86_packed_store(self._vector_arch, acc.dtype, False)
            return store_func(
                PsAddressOf(PsMemAcc(acc.pointer, acc.offset)),
                arg,
            )
        else:
            raise NotImplementedError("Scatter stores not implemented yet.")


@cache
def _x86_packed_load(
    varch: X86VectorArch, vtype: PsVectorType, aligned: bool
) -> CFunction:
    prefix = varch.intrin_prefix(vtype)
    suffix = varch.intrin_suffix(vtype)
    ptr_type = PsPointerType(vtype.scalar_type, const=True)
    return CFunction(
        f"{prefix}_load{'' if aligned else 'u'}_{suffix}", (ptr_type,), vtype
    )


@cache
def _x86_packed_store(
    varch: X86VectorArch, vtype: PsVectorType, aligned: bool
) -> CFunction:
    prefix = varch.intrin_prefix(vtype)
    suffix = varch.intrin_suffix(vtype)
    ptr_type = PsPointerType(vtype.scalar_type, const=True)
    return CFunction(
        f"{prefix}_store{'' if aligned else 'u'}_{suffix}",
        (ptr_type, vtype),
        PsCustomType("void"),
    )


@cache
def _x86_op_intrin(
    varch: X86VectorArch, op: PsUnOp | PsBinOp, vtype: PsVectorType
) -> CFunction:
    prefix = varch.intrin_prefix(vtype)
    suffix = varch.intrin_suffix(vtype)
    rtype = atype = varch.intrin_type(vtype)

    match op:
        case PsVecBroadcast():
            opstr = "set1"
            if vtype.scalar_type == SInt(64) and vtype.vector_entries <= 4:
                suffix += "x"   
            atype = vtype.scalar_type
        case PsAdd():
            opstr = "add"
        case PsSub():
            opstr = "sub"
        case PsMul() if vtype.is_int():
            raise MaterializationError(
                f"Unable to select intrinsic for integer multiplication: "
                f"{varch.name} does not support packed integer multiplication.\n"
                f"    at: {op}"
            )
        case PsMul():
            opstr = "mul"
        case PsDiv():
            opstr = "div"
        case _:
            raise MaterializationError(f"Unable to select operation intrinsic for {type(op)}")

    num_args = 1 if isinstance(op, PsUnOp) else 2
    return CFunction(f"{prefix}_{opstr}_{suffix}", (atype,) * num_args, rtype)
