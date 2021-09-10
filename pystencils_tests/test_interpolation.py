# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
import itertools
from os.path import dirname, join

import numpy as np
import pytest
import sympy

import pystencils
from pystencils.interpolation_astnodes import LinearInterpolator
from pystencils.spatial_coordinates import x_, y_

type_map = {
    np.float32: 'float32',
    np.float64: 'float64',
    np.int32: 'int32',
}

try:
    import pyconrad.autoinit
except Exception:
    import unittest.mock
    pyconrad = unittest.mock.MagicMock()

LENNA_FILE = join(dirname(__file__), 'test_data', 'lenna.png')

try:
    import skimage.io
    lenna = skimage.io.imread(LENNA_FILE, as_gray=True).astype(np.float64)
    pyconrad.imshow(lenna)
except Exception:
    lenna = np.random.rand(20, 30)


def test_interpolation():
    x_f, y_f = pystencils.fields('x,y: float64 [2d]')

    assignments = pystencils.AssignmentCollection({
        y_f.center(): LinearInterpolator(x_f).at([x_ + 2.7, y_ + 7.2])
    })
    print(assignments)
    ast = pystencils.create_kernel(assignments)
    print(ast)
    print(pystencils.show_code(ast))
    kernel = ast.compile()

    pyconrad.imshow(lenna)

    out = np.zeros_like(lenna)
    kernel(x=lenna, y=out)
    pyconrad.imshow(out, "out")


def test_scale_interpolation():
    x_f, y_f = pystencils.fields('x,y: float64 [2d]')

    for address_mode in ['border', 'wrap', 'clamp', 'mirror']:
        assignments = pystencils.AssignmentCollection({
            y_f.center(): LinearInterpolator(x_f, address_mode=address_mode).at([0.5 * x_ + 2.7, 0.25 * y_ + 7.2])
        })
        print(assignments)
        ast = pystencils.create_kernel(assignments)
        print(ast)
        print(pystencils.show_code(ast))
        kernel = ast.compile()

        out = np.zeros_like(lenna)
        kernel(x=lenna, y=out)
        pyconrad.imshow(out, "out " + address_mode)


@pytest.mark.parametrize('address_mode', ['border', 'clamp'])
def test_rotate_interpolation(address_mode):
    """
    'wrap', 'mirror' currently fails on new sympy due to conjugate()
    """
    x_f, y_f = pystencils.fields('x,y: float64 [2d]')

    rotation_angle = sympy.pi / 5

    transformed = sympy.rot_axis3(rotation_angle)[:2, :2] * sympy.Matrix((x_, y_))
    assignments = pystencils.AssignmentCollection({
        y_f.center(): LinearInterpolator(x_f, address_mode=address_mode).at(transformed)
    })
    print(assignments)
    ast = pystencils.create_kernel(assignments)
    print(ast)
    print(pystencils.show_code(ast))
    kernel = ast.compile()

    out = np.zeros_like(lenna)
    kernel(x=lenna, y=out)
    pyconrad.imshow(out, "out " + address_mode)


@pytest.mark.parametrize('dtype', (np.int32, np.float32, np.float64))
@pytest.mark.parametrize('address_mode', ('border', 'wrap', 'clamp', 'mirror'))
@pytest.mark.parametrize('use_textures', ('use_textures', False))
def test_rotate_interpolation_gpu(dtype, address_mode, use_textures):
    pytest.importorskip('pycuda')

    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit  # noqa
    rotation_angle = sympy.pi / 5
    scale = 1

    if dtype == np.int32:
        lenna_gpu = gpuarray.to_gpu(
            np.ascontiguousarray(lenna * 255, dtype))
    else:
        lenna_gpu = gpuarray.to_gpu(
            np.ascontiguousarray(lenna, dtype))
    x_f, y_f = pystencils.fields(f'x,y: {type_map[dtype]} [2d]', ghost_layers=0)

    transformed = scale * \
        sympy.rot_axis3(rotation_angle)[:2, :2] * sympy.Matrix((x_, y_)) - sympy.Matrix([2, 2])
    assignments = pystencils.AssignmentCollection({
        y_f.center(): LinearInterpolator(x_f, address_mode=address_mode).at(transformed)
    })
    print(assignments)
    ast = pystencils.create_kernel(assignments, target=pystencils.Target.GPU,
                                   use_textures_for_interpolation=use_textures)
    print(ast)
    print(pystencils.show_code(ast))
    kernel = ast.compile()

    out = gpuarray.zeros_like(lenna_gpu)
    kernel(x=lenna_gpu, y=out)
    pyconrad.imshow(out,
                    f"out {address_mode} texture:{use_textures} {type_map[dtype]}")


@pytest.mark.parametrize('address_mode', ['border', 'wrap', 'mirror'])
@pytest.mark.parametrize('dtype', [np.float64, np.float32, np.int32])
@pytest.mark.parametrize('use_textures', ('use_textures', False,))
def test_shift_interpolation_gpu(address_mode, dtype, use_textures):
    sver = sympy.__version__.split(".")
    if (int(sver[0]) == 1 and int(sver[1]) < 2) and address_mode == 'mirror':
        pytest.skip("% printed as fmod on old sympy")
    pytest.importorskip('pycuda')

    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit  # noqa

    rotation_angle = 0  # sympy.pi / 5
    scale = 1
    # shift = - sympy.Matrix([1.5, 1.5])
    shift = sympy.Matrix((0.0, 0.0))

    if dtype == np.int32:
        lenna_gpu = gpuarray.to_gpu(
            np.ascontiguousarray(lenna * 255, dtype))
    else:
        lenna_gpu = gpuarray.to_gpu(
            np.ascontiguousarray(lenna, dtype))

    x_f, y_f = pystencils.fields(f'x,y: {type_map[dtype]} [2d]', ghost_layers=0)

    if use_textures:
        transformed = scale * sympy.rot_axis3(rotation_angle)[:2, :2] * sympy.Matrix((x_, y_)) + shift
    else:
        transformed = scale * sympy.rot_axis3(rotation_angle)[:2, :2] * sympy.Matrix((x_, y_)) + shift
    assignments = pystencils.AssignmentCollection({
        y_f.center(): LinearInterpolator(x_f, address_mode=address_mode).at(transformed)
    })
    # print(assignments)
    ast = pystencils.create_kernel(assignments, target=pystencils.Target.GPU,
                                   use_textures_for_interpolation=use_textures)
    # print(ast)
    print(pystencils.show_code(ast))
    kernel = ast.compile()

    out = gpuarray.zeros_like(lenna_gpu)
    kernel(x=lenna_gpu, y=out)
    pyconrad.imshow(out,
                    f"out {address_mode} texture:{use_textures} {type_map[dtype]}")


@pytest.mark.parametrize('address_mode', ['border', 'clamp'])
def test_rotate_interpolation_size_change(address_mode):
    """
    'wrap', 'mirror' currently fails on new sympy due to conjugate()
    """
    x_f, y_f = pystencils.fields('x,y: float64 [2d]')

    rotation_angle = sympy.pi / 5

    transformed = sympy.rot_axis3(rotation_angle)[:2, :2] * sympy.Matrix((x_, y_))
    assignments = pystencils.AssignmentCollection({
        y_f.center(): LinearInterpolator(x_f, address_mode=address_mode).at(transformed)
    })
    print(assignments)
    ast = pystencils.create_kernel(assignments)
    print(ast)
    print(pystencils.show_code(ast))
    kernel = ast.compile()

    out = np.zeros((100, 150), np.float64)
    kernel(x=lenna, y=out)
    pyconrad.imshow(out, "small out " + address_mode)


@pytest.mark.parametrize('address_mode, target',
                         itertools.product(['border', 'wrap', 'clamp', 'mirror'], [pystencils.Target.CPU]))
def test_field_interpolated(address_mode, target):
    x_f, y_f = pystencils.fields('x,y: float64 [2d]')

    assignments = pystencils.AssignmentCollection({
        y_f.center(): x_f.interpolated_access([0.5 * x_ + 2.7, 0.25 * y_ + 7.2], address_mode=address_mode)
    })
    print(assignments)
    ast = pystencils.create_kernel(assignments, target=target)
    print(ast)
    print(pystencils.show_code(ast))
    kernel = ast.compile()

    out = np.zeros_like(lenna)
    kernel(x=lenna, y=out)
    pyconrad.imshow(out, "out " + address_mode)


def test_spatial_derivative():
    x, y = pystencils.fields('x, y:  float32[2d]')
    tx, ty = pystencils.fields('t_x, t_y: float32[2d]')

    diff = sympy.diff(x.interpolated_access((tx.center, ty.center)), tx.center)
    print("diff: " + str(diff))
