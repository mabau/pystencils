import numpy as np
from itertools import product

from pystencils import CreateKernelConfig, create_kernel
from pystencils import Assignment, Field, Target
from pystencils.slicing import get_periodic_boundary_src_dst_slices, normalize_slice


def create_copy_kernel(
    domain_size,
    src_slice,
    dst_slice,
    index_dimensions=0,
    index_dim_shape=1,
    dtype=np.float64,
):
    """Copies a rectangular part of an array to another non-overlapping part"""

    field = Field.create_generic(
        "field", len(domain_size), index_dimensions=index_dimensions, dtype=dtype
    )
    normalized_src_slice = normalize_slice(src_slice, field.spatial_shape)
    normalized_dst_slice = normalize_slice(dst_slice, field.spatial_shape)

    offset = [
        s1.start - s2.start
        for s1, s2 in zip(normalized_src_slice, normalized_dst_slice)
    ]
    assert offset == [
        s1.stop - s2.stop for s1, s2 in zip(normalized_src_slice, normalized_dst_slice)
    ], "Slices have to have same size"

    update_eqs = []
    if index_dimensions < 2:
        index_dim_shape = [index_dim_shape]
    for i in product(*[range(d) for d in index_dim_shape]):
        eq = Assignment(field(*i), field[tuple(offset)](*i))
        update_eqs.append(eq)

    config = CreateKernelConfig(
        target=Target.GPU, iteration_slice=dst_slice, skip_independence_check=True
    )

    ast = create_kernel(update_eqs, config=config)
    return ast


def get_periodic_boundary_functor(
    stencil,
    domain_size,
    index_dimensions=0,
    index_dim_shape=1,
    ghost_layers=1,
    thickness=None,
    dtype=np.float64,
    target=Target.GPU,
):
    assert target in {Target.GPU}
    src_dst_slice_tuples = get_periodic_boundary_src_dst_slices(
        stencil, ghost_layers, thickness
    )
    kernels = []

    for src_slice, dst_slice in src_dst_slice_tuples:
        ast = create_copy_kernel(
            domain_size, src_slice, dst_slice, index_dimensions, index_dim_shape, dtype
        )
        kernels.append(ast.compile())

    def functor(field, **_):
        for kernel in kernels:
            kernel(field=field)

    return functor
