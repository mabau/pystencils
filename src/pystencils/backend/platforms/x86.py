from __future__ import annotations
from enum import Enum
from functools import cache
from typing import Sequence

from pymbolic.primitives import Expression

from ..arrays import PsVectorArrayAccess
from ..transformations.vector_intrinsics import IntrinsicOps
from ..typed_expressions import PsTypedConstant
from ..types import PsCustomType, PsVectorType
from ..functions import address_of

from .generic_cpu import GenericVectorCpu, IntrinsicsError

from ..types.quick import Fp, SInt
from ..functions import CFunction


class X86VectorArch(Enum):
    SSE = 128
    AVX = 256
    AVX512 = 512

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
                raise IntrinsicsError(
                    f"X86/{self} does not support vector width {other}"
                )

        return prefix

    def intrin_suffix(self, vtype: PsVectorType) -> str:
        scalar_type = vtype.scalar_type
        match scalar_type:
            case Fp(16) if self >= X86VectorArch.AVX512:
                suffix = "ph"
            case Fp(32):
                suffix = "ps"
            case Fp(64):
                suffix = "pd"
            case SInt(width):
                suffix = f"epi{width}"
            case _:
                raise IntrinsicsError(
                    f"X86/{self} does not support scalar type {scalar_type}"
                )

        return suffix


class X86VectorCpu(GenericVectorCpu):
    """Platform modelling the X86 SSE/AVX/AVX512 vector architectures.

    All intrinsics information is extracted from
    https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html.
    """

    def __init__(self, vector_arch: X86VectorArch):
        self._vector_arch = vector_arch

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
        scalar_type = vector_type.scalar_type
        match scalar_type:
            case Fp(16) if self._vector_arch >= X86VectorArch.AVX512:
                suffix = "h"
            case Fp(32):
                suffix = ""
            case Fp(64):
                suffix = "d"
            case SInt(_):
                suffix = "i"
            case _:
                raise IntrinsicsError(
                    f"X86/{self._vector_arch} does not support scalar type {scalar_type}"
                )

        if vector_type.width > self._vector_arch.max_vector_width:
            raise IntrinsicsError(
                f"X86/{self._vector_arch} does not support {vector_type}"
            )
        return PsCustomType(f"__m{vector_type.width}{suffix}")

    def constant_vector(self, c: PsTypedConstant) -> Expression:
        vtype = c.dtype
        assert isinstance(vtype, PsVectorType)

        prefix = self._vector_arch.intrin_prefix(vtype)
        suffix = self._vector_arch.intrin_suffix(vtype)
        set_func = CFunction(f"{prefix}_set_{suffix}", vtype.vector_entries)

        values = c.value
        return set_func(*values)

    def op_intrinsic(
        self, op: IntrinsicOps, vtype: PsVectorType, args: Sequence[Expression]
    ) -> Expression:
        func = _x86_op_intrin(self._vector_arch, op, vtype)
        return func(*args)

    def vector_load(self, acc: PsVectorArrayAccess) -> Expression:
        if acc.stride == 1:
            load_func = _x86_packed_load(self._vector_arch, acc.dtype, False)
            return load_func(address_of(acc.base_ptr[acc.base_index]))
        else:
            raise NotImplementedError("Gather loads not implemented yet.")

    def vector_store(self, acc: PsVectorArrayAccess, arg: Expression) -> Expression:
        if acc.stride == 1:
            store_func = _x86_packed_store(self._vector_arch, acc.dtype, False)
            return store_func(address_of(acc.base_ptr[acc.base_index]), arg)
        else:
            raise NotImplementedError("Scatter stores not implemented yet.")


@cache
def _x86_packed_load(
    varch: X86VectorArch, vtype: PsVectorType, aligned: bool
) -> CFunction:
    prefix = varch.intrin_prefix(vtype)
    suffix = varch.intrin_suffix(vtype)
    return CFunction(f"{prefix}_load{'' if aligned else 'u'}_{suffix}", 1)


@cache
def _x86_packed_store(
    varch: X86VectorArch, vtype: PsVectorType, aligned: bool
) -> CFunction:
    prefix = varch.intrin_prefix(vtype)
    suffix = varch.intrin_suffix(vtype)
    return CFunction(f"{prefix}_store{'' if aligned else 'u'}_{suffix}", 2)


@cache
def _x86_op_intrin(
    varch: X86VectorArch, op: IntrinsicOps, vtype: PsVectorType
) -> CFunction:
    prefix = varch.intrin_prefix(vtype)
    suffix = varch.intrin_suffix(vtype)

    match op:
        case IntrinsicOps.ADD:
            opstr = "add"
        case IntrinsicOps.SUB:
            opstr = "sub"
        case IntrinsicOps.MUL:
            opstr = "mul"
        case IntrinsicOps.DIV:
            opstr = "div"
        case IntrinsicOps.FMA:
            opstr = "fmadd"
        case _:
            assert False

    return CFunction(f"{prefix}_{opstr}_{suffix}", 3 if op == IntrinsicOps.FMA else 2)
