import pytest

from dataclasses import dataclass
from pystencils.backend.memory import PsSymbol, PsSymbolProperty, UniqueSymbolProperty


def test_properties():
    @dataclass(frozen=True)
    class NumbersProperty(PsSymbolProperty):
        n: int
        x: float

    @dataclass(frozen=True)
    class StringProperty(PsSymbolProperty):
        s: str

    @dataclass(frozen=True)
    class MyUniqueProperty(UniqueSymbolProperty):
        val: int

    s = PsSymbol("s")

    assert not s.properties

    s.add_property(NumbersProperty(42, 8.71))
    assert s.properties == {NumbersProperty(42, 8.71)}

    #   no duplicates
    s.add_property(NumbersProperty(42, 8.71))
    assert s.properties == {NumbersProperty(42, 8.71)}

    s.add_property(StringProperty("pystencils"))
    assert s.properties == {NumbersProperty(42, 8.71), StringProperty("pystencils")}

    assert s.get_properties(NumbersProperty) == {NumbersProperty(42, 8.71)}
    
    assert not s.get_properties(MyUniqueProperty)
    
    s.add_property(MyUniqueProperty(13))
    assert s.get_properties(MyUniqueProperty) == {MyUniqueProperty(13)}

    #   Adding the same one again does not raise
    s.add_property(MyUniqueProperty(13))
    assert s.get_properties(MyUniqueProperty) == {MyUniqueProperty(13)}

    with pytest.raises(ValueError):
        s.add_property(MyUniqueProperty(14))

    s.remove_property(MyUniqueProperty(13))
    assert not s.get_properties(MyUniqueProperty)
