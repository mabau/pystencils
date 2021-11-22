import numpy as np
from itertools import product

import pystencils.gpucuda
from pystencils import Assignment, Field
from pystencils.gpucuda.kernelcreation import create_cuda_kernel
from pystencils.enums import Target
from pystencils.slicing import get_periodic_boundary_src_dst_slices, normalize_slice


def create_copy_kernel(domain_size, from_slice, to_slice, index_dimensions=0, index_dim_shape=1, dtype=np.float64):
    """Copies a rectangular part of an array to another non-overlapping part"""

    f = Field.create_generic("pdfs", len(domain_size), index_dimensions=index_dimensions, dtype=dtype)
    normalized_from_slice = normalize_slice(from_slice, f.spatial_shape)
    normalized_to_slice = normalize_slice(to_slice, f.spatial_shape)

    offset = [s1.start - s2.start for s1, s2 in zip(normalized_from_slice, normalized_to_slice)]
    assert offset == [s1.stop - s2.stop for s1, s2 in zip(normalized_from_slice, normalized_to_slice)], \
        "Slices have to have same size"

    update_eqs = []
    if index_dimensions < 2:
        index_dim_shape = [index_dim_shape]
    for i in product(*[range(d) for d in index_dim_shape]):
        eq = Assignment(f(*i), f[tuple(offset)](*i))
        update_eqs.append(eq)

    ast = create_cuda_kernel(update_eqs, iteration_slice=to_slice, skip_independence_check=True)
    return ast


def get_periodic_boundary_functor(stencil, domain_size, index_dimensions=0, index_dim_shape=1, ghost_layers=1,
                                  thickness=None, dtype=float, target=Target.GPU):
    assert target in {Target.GPU}
    src_dst_slice_tuples = get_periodic_boundary_src_dst_slices(stencil, ghost_layers, thickness)
    kernels = []

    for src_slice, dst_slice in src_dst_slice_tuples:
        ast = create_copy_kernel(domain_size, src_slice, dst_slice, index_dimensions, index_dim_shape, dtype)
        kernels.append(pystencils.gpucuda.make_python_function(ast))

    def functor(pdfs, **_):
        for kernel in kernels:
            kernel(pdfs=pdfs)

    return functor
