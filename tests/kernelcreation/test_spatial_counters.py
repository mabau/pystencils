import pytest
import numpy as np

from pystencils import (
    Field,
    Assignment,
    create_kernel,
    CreateKernelConfig,
    DEFAULTS,
    FieldType,
)
from pystencils.sympyextensions import CastFunc


@pytest.mark.parametrize("index_dtype", ["int16", "int32", "uint32", "int64"])
def test_spatial_counters_dense(index_dtype):
    #   Parametrized over index_dtype to make sure the `DynamicType.INDEX` in the
    #   DEFAULTS works validly
    x, y, z = DEFAULTS.spatial_counters

    f = Field.create_generic("f", 3, "float64", index_shape=(3,), layout="fzyx")

    asms = [
        Assignment(f(0), CastFunc.as_numeric(z)),
        Assignment(f(1), CastFunc.as_numeric(y)),
        Assignment(f(2), CastFunc.as_numeric(x)),
    ]

    cfg = CreateKernelConfig(index_dtype=index_dtype)
    kernel = create_kernel(asms, cfg).compile()

    f_arr = np.zeros((16, 16, 16, 3))
    kernel(f=f_arr)

    expected = np.mgrid[0:16, 0:16, 0:16].astype(np.float64).transpose()

    np.testing.assert_equal(f_arr, expected)


@pytest.mark.parametrize("index_dtype", ["int16", "int32", "uint32", "int64"])
def test_spatial_counters_sparse(index_dtype):
    x, y, z = DEFAULTS.spatial_counters

    f = Field.create_generic("f", 3, "float64", index_shape=(3,), layout="fzyx")

    asms = [
        Assignment(f(0), CastFunc.as_numeric(x)),
        Assignment(f(1), CastFunc.as_numeric(y)),
        Assignment(f(2), CastFunc.as_numeric(z)),
    ]

    idx_struct = DEFAULTS.index_struct(index_dtype, 3)
    idx_field = Field.create_generic(
        "index", 1, idx_struct, field_type=FieldType.INDEXED
    )

    cfg = CreateKernelConfig(index_dtype=index_dtype, index_field=idx_field)
    kernel = create_kernel(asms, cfg).compile()

    f_arr = np.zeros((16, 16, 16, 3))
    idx_arr = np.array(
        [(1, 4, 3), (5, 1, 6), (9, 5, 1), (3, 13, 7)], dtype=idx_struct.numpy_dtype
    )

    kernel(f=f_arr, index=idx_arr)

    for t in idx_arr:
        assert f_arr[t[0], t[1], t[2], 0] == t[0].astype(np.float64)
        assert f_arr[t[0], t[1], t[2], 1] == t[1].astype(np.float64)
        assert f_arr[t[0], t[1], t[2], 2] == t[2].astype(np.float64)
