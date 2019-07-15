import numpy as np

from pystencils import Assignment, Field
from pystencils.gpucuda import make_python_function
from pystencils.gpucuda.kernelcreation import create_cuda_kernel
from pystencils.slicing import get_periodic_boundary_src_dst_slices, normalize_slice


def create_copy_kernel(domain_size, from_slice, to_slice, index_dimensions=0, index_dim_shape=1, dtype=np.float64):
    """Copies a rectangular part of an array to another non-overlapping part"""
    if index_dimensions not in (0, 1):
        raise NotImplementedError("Works only for one or zero index coordinates")

    f = Field.create_generic("pdfs", len(domain_size), index_dimensions=index_dimensions, dtype=dtype)
    normalized_from_slice = normalize_slice(from_slice, f.spatial_shape)
    normalized_to_slice = normalize_slice(to_slice, f.spatial_shape)

    offset = [s1.start - s2.start for s1, s2 in zip(normalized_from_slice, normalized_to_slice)]
    assert offset == [s1.stop - s2.stop for s1, s2 in zip(normalized_from_slice, normalized_to_slice)], \
        "Slices have to have same size"

    update_eqs = []
    for i in range(index_dim_shape):
        eq = Assignment(f(i), f[tuple(offset)](i))
        update_eqs.append(eq)

    ast = create_cuda_kernel(update_eqs, iteration_slice=to_slice, skip_independence_check=True)
    return make_python_function(ast)


def get_periodic_boundary_functor(stencil, domain_size, index_dimensions=0, index_dim_shape=1, ghost_layers=1,
                                  thickness=None, dtype=float):
    src_dst_slice_tuples = get_periodic_boundary_src_dst_slices(stencil, ghost_layers, thickness)
    kernels = []
    index_dimensions = index_dimensions
    for src_slice, dst_slice in src_dst_slice_tuples:
        kernels.append(create_copy_kernel(domain_size, src_slice, dst_slice, index_dimensions, index_dim_shape, dtype))

    def functor(pdfs, **_):
        for kernel in kernels:
            kernel(pdfs=pdfs)

    return functor
