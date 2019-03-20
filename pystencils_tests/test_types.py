from pystencils import data_types
from pystencils.data_types import *


def test_parsing():
    assert str(data_types.create_composite_type_from_string("const double *")) == "double const *"
    assert str(data_types.create_composite_type_from_string("double const *")) == "double const *"

    t1 = data_types.create_composite_type_from_string("const double * const * const restrict")
    t2 = data_types.create_composite_type_from_string(str(t1))
    assert t1 == t2


def test_collation():
    double_type = create_type("double")
    float_type = create_type("float32")
    double4_type = VectorType(double_type, 4)
    float4_type = VectorType(float_type, 4)
    assert collate_types([double_type, float_type]) == double_type
    assert collate_types([double4_type, float_type]) == double4_type
    assert collate_types([double4_type, float4_type]) == double4_type
