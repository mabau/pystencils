from copy import copy, deepcopy

from pystencils.field import Field
from pystencils.typing import TypedSymbol


def test_field_access():
    field = Field.create_generic('some_field', spatial_dimensions=2, index_dimensions=0)
    copy(field(0))
    field_copy = deepcopy(field(0))
    assert field_copy.field.spatial_dimensions == 2


def test_typed_symbol():
    ts = TypedSymbol("s", "double")
    copy(ts)
    ts_copy = deepcopy(ts)
    assert str(ts_copy.dtype).strip() == "double"
