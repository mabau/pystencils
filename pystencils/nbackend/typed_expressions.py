from __future__ import annotations

from typing import TypeAlias, Union, Any

import pymbolic.primitives as pb

from ..typing import AbstractType, BasicType

class PsTypedSymbol(pb.Variable):
    def __init__(self, name: str, dtype: AbstractType):
        super(PsTypedSymbol, self).__init__(name)
        self._dtype = dtype

    @property
    def dtype(self) -> AbstractType:
        return self._dtype


class PsArrayBasePointer(PsTypedSymbol):
    def __init__(self, name: str, base_type: AbstractType):
        super(PsArrayBasePointer, self).__init__(name, base_type)


class PsArrayAccess(pb.Subscript):
    def __init__(self, base_ptr: PsArrayBasePointer, index: pb.Expression):
        super(PsArrayAccess, self).__init__(base_ptr, index)


PsLvalue : TypeAlias = Union[PsTypedSymbol, PsArrayAccess]


class PsTypedConstant:

    @staticmethod
    def _cast(value, target_dtype: AbstractType):
        if isinstance(value, PsTypedConstant):
            if value._dtype != target_dtype:
                raise ValueError(f"Incompatible types: {value._dtype} and {target_dtype}")
            return value
        
        # TODO check legality
        return PsTypedConstant(value, target_dtype)

    def __init__(self, value, dtype: AbstractType):
        """Represents typed constants occuring in the pystencils AST"""
        if isinstance(dtype, BasicType):
            dtype = BasicType(dtype, const = True)
            self._value = dtype.numpy_dtype.type(value)
        else:
            raise ValueError(f"Cannot create constant of type {dtype}")
        
        self._dtype = dtype

    def __str__(self) -> str:
        return str(self._value)
    
    def __add__(self, other: Any):
        other = PsTypedConstant._cast(other, self._dtype)
        
        return PsTypedConstant(self._value + other._value, self._dtype)

    def __mul__(self, other: Any):
        other = PsTypedConstant._cast(other, self._dtype)
        
        return PsTypedConstant(self._value * other._value, self._dtype)
    
    def __sub__(self, other: Any):
        other = PsTypedConstant._cast(other, self._dtype)
        
        return PsTypedConstant(self._value - other._value, self._dtype)

    # TODO: Remaining operators


pb.VALID_CONSTANT_CLASSES += (PsTypedConstant,)
