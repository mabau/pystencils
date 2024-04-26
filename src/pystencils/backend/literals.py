from __future__ import annotations
from ..types import PsType, constify


class PsLiteral:
    """Representation of literal code.
    
    Instances of this class represent code literals inside the AST.
    These literals are not to be confused with C literals; the name `Literal` refers to the fact that
    the code generator takes them "literally", printing them as they are.

    Each literal has to be annotated with a type, and is considered constant within the scope of a kernel.
    Instances of `PsLiteral` are immutable.
    """

    __match_args__ = ("text", "dtype")

    def __init__(self, text: str, dtype: PsType) -> None:
        self._text = text
        self._dtype = constify(dtype)

    @property
    def text(self) -> str:
        return self._text
    
    @property
    def dtype(self) -> PsType:
        return self._dtype
    
    def __str__(self) -> str:
        return f"{self._text}: {self._dtype}"
    
    def __repr__(self) -> str:
        return f"PsLiteral({repr(self._text)}, {repr(self._dtype)})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PsLiteral):
            return False
        
        return self._text == other._text and self._dtype == other._dtype
    
    def __hash__(self) -> int:
        return hash((PsLiteral, self._text, self._dtype))
