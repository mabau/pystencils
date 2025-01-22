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
    PsConstantExpr,
    PsCast,
    PsCall,
)
from ..ast.vector import PsVecMemAcc, PsVecBroadcast
from ...types import PsCustomType, PsVectorType, PsPointerType
from ..constants import PsConstant

from ..exceptions import MaterializationError
from .generic_cpu import GenericVectorCpu
from ..kernelcreation import KernelCreationContext

from ...types.quick import Fp, UInt, SInt
from ..functions import CFunction, PsMathFunction, MathFunctions


class X86VectorArch(Enum):
    SSE = 128
    AVX = 256
    AVX512 = 512
    AVX512_FP16 = AVX512 + 1  # TODO improve modelling?

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
            raise MaterializationError(f"x86/{self} does not support {vtype}")
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
                intrinsic = func(*operands)
                intrinsic.dtype = func.return_type
                return intrinsic
            case _:
                raise MaterializationError(f"Cannot map {type(expr)} to x86 intrinsic")

    def math_func_intrinsic(
        self, expr: PsCall, operands: Sequence[PsExpression]
    ) -> PsExpression:
        assert isinstance(expr.function, PsMathFunction)

        vtype = expr.get_dtype()
        assert isinstance(vtype, PsVectorType)

        prefix = self._vector_arch.intrin_prefix(vtype)
        suffix = self._vector_arch.intrin_suffix(vtype)
        rtype = atype = self._vector_arch.intrin_type(vtype)

        match expr.function.func:
            case (
                MathFunctions.Exp
                | MathFunctions.Log
                | MathFunctions.Sin
                | MathFunctions.Cos
                | MathFunctions.Tan
                | MathFunctions.Sinh
                | MathFunctions.Cosh
                | MathFunctions.ASin
                | MathFunctions.ACos
                | MathFunctions.ATan
                | MathFunctions.ATan2
                | MathFunctions.Pow
            ):
                raise MaterializationError(
                    "Trigonometry, exp, log, and pow require SVML."
                )

            case MathFunctions.Floor | MathFunctions.Ceil if vtype.is_float():
                opstr = expr.function.func.function_name
                if vtype.width > 256:
                    raise MaterializationError("512bit ceil/floor require SVML.")

            case MathFunctions.Min | MathFunctions.Max:
                opstr = expr.function.func.function_name
                if (
                    vtype.is_int()
                    and vtype.scalar_type.width == 64
                    and self._vector_arch < X86VectorArch.AVX512
                ):
                    raise MaterializationError(
                        "64bit integer (signed and unsigned) min/max intrinsics require AVX512."
                    )

            case MathFunctions.Abs:
                assert len(operands) == 1, "abs takes exactly one argument."
                op = operands[0]

                match vtype.scalar_type:
                    case UInt():
                        return op

                    case SInt(width):
                        opstr = expr.function.func.function_name
                        if width == 64 and self._vector_arch < X86VectorArch.AVX512:
                            raise MaterializationError(
                                "64bit integer abs intrinsic requires AVX512."
                            )

                    case Fp():
                        neg_zero = self.constant_intrinsic(PsConstant(-0.0, vtype))

                        opstr = "andnot"
                        func = CFunction(
                            f"{prefix}_{opstr}_{suffix}", (atype,) * 2, rtype
                        )

                        return func(neg_zero, op)

            case _:
                raise MaterializationError(
                    f"x86/{self} does not support {expr.function.func.function_name} on type {vtype}."
                )

        if expr.function.func in [
            MathFunctions.ATan2,
            MathFunctions.Min,
            MathFunctions.Max,
        ]:
            num_args = 2
        else:
            num_args = 1

        func = CFunction(f"{prefix}_{opstr}_{suffix}", (atype,) * num_args, rtype)
        return func(*operands)

    def vector_load(self, acc: PsVecMemAcc) -> PsExpression:
        if acc.stride is None:
            load_func, addr_type = _x86_packed_load(self._vector_arch, acc.dtype, False)
            addr: PsExpression = PsAddressOf(PsMemAcc(acc.pointer, acc.offset))
            if addr_type:
                addr = PsCast(addr_type, addr)
            intrinsic = load_func(addr)
            intrinsic.dtype = load_func.return_type
            return intrinsic
        else:
            raise NotImplementedError("Gather loads not implemented yet.")

    def vector_store(self, acc: PsVecMemAcc, arg: PsExpression) -> PsExpression:
        if acc.stride is None:
            store_func, addr_type = _x86_packed_store(
                self._vector_arch, acc.dtype, False
            )
            addr: PsExpression = PsAddressOf(PsMemAcc(acc.pointer, acc.offset))
            if addr_type:
                addr = PsCast(addr_type, addr)
            intrinsic = store_func(addr, arg)
            intrinsic.dtype = store_func.return_type
            return intrinsic
        else:
            raise NotImplementedError("Scatter stores not implemented yet.")


@cache
def _x86_packed_load(
    varch: X86VectorArch, vtype: PsVectorType, aligned: bool
) -> tuple[CFunction, PsPointerType | None]:
    prefix = varch.intrin_prefix(vtype)
    ptr_type = PsPointerType(vtype.scalar_type, const=True)

    if isinstance(vtype.scalar_type, SInt):
        suffix = f"si{vtype.width}"
        addr_type = PsPointerType(varch.intrin_type(vtype))
    else:
        suffix = varch.intrin_suffix(vtype)
        addr_type = None

    return (
        CFunction(
            f"{prefix}_load{'' if aligned else 'u'}_{suffix}", (ptr_type,), vtype
        ),
        addr_type,
    )


@cache
def _x86_packed_store(
    varch: X86VectorArch, vtype: PsVectorType, aligned: bool
) -> tuple[CFunction, PsPointerType | None]:
    prefix = varch.intrin_prefix(vtype)
    ptr_type = PsPointerType(vtype.scalar_type, const=True)

    if isinstance(vtype.scalar_type, SInt):
        suffix = f"si{vtype.width}"
        addr_type = PsPointerType(varch.intrin_type(vtype))
    else:
        suffix = varch.intrin_suffix(vtype)
        addr_type = None

    return (
        CFunction(
            f"{prefix}_store{'' if aligned else 'u'}_{suffix}",
            (ptr_type, vtype),
            PsCustomType("void"),
        ),
        addr_type,
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
        case PsCast(target_type, arg):
            atype = arg.dtype
            widest_type = max(vtype, atype, key=lambda t: t.width)

            assert target_type == vtype, "type mismatch"
            assert isinstance(atype, PsVectorType)

            def panic(detail: str = ""):
                raise MaterializationError(
                    f"Unable to select intrinsic for type conversion: "
                    f"{varch.name} does not support packed conversion from {atype} to {target_type}. {detail}\n"
                    f"    at: {op}"
                )

            if atype == vtype:
                panic("Use `EliminateConstants` to eliminate trivial casts.")

            match (atype.scalar_type, vtype.scalar_type):
                # Not supported: cvtepi8_pX, cvtpX_epi8, cvtepi16_p[sd], cvtp[sd]_epi16
                case (
                    (SInt(8), Fp())
                    | (Fp(), SInt(8))
                    | (SInt(16), Fp(32))
                    | (SInt(16), Fp(64))
                    | (Fp(32), SInt(16))
                    | (Fp(64), SInt(16))
                ):
                    panic()
                # AVX512 only: cvtepi64_pX, cvtpX_epi64
                case (SInt(64), Fp()) | (
                    Fp(),
                    SInt(64),
                ) if varch < X86VectorArch.AVX512:
                    panic()
                # AVX512 only: cvtepiA_epiT if A > T
                case (SInt(a), SInt(t)) if a > t and varch < X86VectorArch.AVX512:
                    panic()

                case _:
                    prefix = varch.intrin_prefix(widest_type)
                    opstr = f"cvt{varch.intrin_suffix(atype)}"

        case _:
            raise MaterializationError(
                f"Unable to select operation intrinsic for {type(op)}"
            )

    num_args = 1 if isinstance(op, PsUnOp) else 2
    return CFunction(f"{prefix}_{opstr}_{suffix}", (atype,) * num_args, rtype)
