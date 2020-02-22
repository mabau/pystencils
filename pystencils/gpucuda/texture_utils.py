# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""

from os.path import dirname, isdir, join
from typing import Union

import numpy as np

try:
    import pycuda.driver as cuda
    from pycuda import gpuarray
    import pycuda
except Exception:
    pass


def pow_two_divider(n):
    if n == 0:
        return 0
    divider = 1
    while (n & divider) == 0:
        divider <<= 1
    return divider


def ndarray_to_tex(tex_ref,  # type: Union[cuda.TextureReference, cuda.SurfaceReference]
                   ndarray,
                   address_mode=None,
                   filter_mode=None,
                   use_normalized_coordinates=False,
                   read_as_integer=False):

    if isinstance(address_mode, str):
        address_mode = getattr(pycuda.driver.address_mode, address_mode.upper())
    if address_mode is None:
        address_mode = cuda.address_mode.BORDER
    if filter_mode is None:
        filter_mode = cuda.filter_mode.LINEAR

    if isinstance(ndarray, np.ndarray):
        cu_array = cuda.np_to_array(ndarray, 'C')
    elif isinstance(ndarray, gpuarray.GPUArray):
        cu_array = cuda.gpuarray_to_array(ndarray, 'C')
    else:
        raise TypeError(
            'ndarray must be numpy.ndarray or pycuda.gpuarray.GPUArray')

    tex_ref.set_array(cu_array)

    tex_ref.set_address_mode(0, address_mode)
    if ndarray.ndim >= 2:
        tex_ref.set_address_mode(1, address_mode)
    if ndarray.ndim >= 3:
        tex_ref.set_address_mode(2, address_mode)
    tex_ref.set_filter_mode(filter_mode)

    if not use_normalized_coordinates:
        tex_ref.set_flags(tex_ref.get_flags() & ~cuda.TRSF_NORMALIZED_COORDINATES)

    if not read_as_integer:
        tex_ref.set_flags(tex_ref.get_flags() & ~cuda.TRSF_READ_AS_INTEGER)


def prefilter_for_cubic_bspline(gpuarray):
    import pycuda.autoinit  # NOQA
    from pycuda.compiler import SourceModule

    ndim = gpuarray.ndim
    assert ndim == 2 or ndim == 3, "Only 2d or 3d supported"
    assert isdir(join(dirname(__file__), "CubicInterpolationCUDA", "code")), \
        "Submodule CubicInterpolationCUDA does not exist"
    nvcc_options = ["-w", "-std=c++11", "-Wno-deprecated-gpu-targets"]
    nvcc_options += ["-I" + join(dirname(__file__), "CubicInterpolationCUDA", "code")]
    nvcc_options += ["-I" + join(dirname(__file__), "CubicInterpolationCUDA", "code", "internal")]

    file_name = join(dirname(__file__), "CubicInterpolationCUDA", "code", "cubicPrefilter%iD.cu" % ndim)
    with open(file_name) as file:
        code = file.read()

    mod = SourceModule(code, options=nvcc_options)

    if ndim == 2:
        height, width = gpuarray.shape
        block = min(pow_two_divider(height), 64)
        grid = height // block
        func = mod.get_function('SamplesToCoefficients2DXf')
        func(gpuarray, np.uint32(gpuarray.strides[-2]), *(np.uint32(r)
                                                          for r in reversed(gpuarray.shape)),
             block=(block, 1, 1),
             grid=(grid, 1, 1))

        block = min(pow_two_divider(width), 64)
        grid = width // block
        func = mod.get_function('SamplesToCoefficients2DYf')
        func(gpuarray, np.uint32(gpuarray.strides[-2]), *(np.uint32(r)
                                                          for r in reversed(gpuarray.shape)),
             block=(block, 1, 1),
             grid=(grid, 1, 1))
    elif ndim == 3:
        depth, height, width = gpuarray.shape
        dimX = min(min(pow_two_divider(width), pow_two_divider(height)), 64)
        dimY = min(min(pow_two_divider(depth), pow_two_divider(height)), 512 / dimX)
        block = (dimX, dimY, 1)

        dimGridX = (height // block[0], depth // block[1], 1)
        dimGridY = (width // block[0], depth // block[1], 1)
        dimGridZ = (width // block[0], height // block[1], 1)

        func = mod.get_function("SamplesToCoefficients3DXf")
        func(gpuarray, np.uint32(gpuarray.strides[-2]), *(np.uint32(r)
                                                          for r in reversed(gpuarray.shape)),
             block=block,
             grid=dimGridX)
        func = mod.get_function("SamplesToCoefficients3DYf")
        func(gpuarray, np.uint32(gpuarray.strides[-2]), *(np.uint32(r)
                                                          for r in reversed(gpuarray.shape)),
             block=block,
             grid=dimGridY)
        func = mod.get_function("SamplesToCoefficients3DZf")
        func(gpuarray, np.uint32(gpuarray.strides[-2]), *(np.uint32(r)
                                                          for r in reversed(gpuarray.shape)),
             block=block,
             grid=dimGridZ)
