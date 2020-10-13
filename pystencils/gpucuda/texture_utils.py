# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
from typing import Union

import numpy as np

try:
    import pycuda.driver as cuda
    from pycuda import gpuarray
    import pycuda
except Exception:
    pass


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
