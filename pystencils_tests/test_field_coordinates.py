# -*- coding: utf-8 -*-
#
# Copyright © 2019 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""
from os.path import dirname, join

import numpy as np
import sympy

import pystencils
from pystencils.interpolation_astnodes import LinearInterpolator

try:
    import pyconrad.autoinit
except Exception:
    import unittest.mock
    pyconrad = unittest.mock.MagicMock()

LENNA_FILE = join(dirname(__file__), 'test_data', 'lenna.png')

try:
    import skimage.io
    lenna = skimage.io.imread(LENNA_FILE, as_gray=True).astype(np.float32)
except Exception:
    lenna = np.random.rand(20, 30).astype(np.float32)


def test_rotate_center():
    x, y = pystencils.fields('x, y:  float32[2d]')

    # Rotate around center when setting coordindates origins to field centers
    x.set_coordinate_origin_to_field_center()
    y.set_coordinate_origin_to_field_center()

    rotation_angle = sympy.pi / 5
    transform_matrix = sympy.rot_axis3(rotation_angle)[:2, :2]

    # Generic matrix transform works like that (for rotation it would be more clever to use transform_matrix.T)
    inverse_matrix = transform_matrix.inv()
    input_coordinate = x.physical_to_index(inverse_matrix @ y.physical_coordinates)

    assignments = pystencils.AssignmentCollection({
        y.center(): LinearInterpolator(x).at(input_coordinate)
    })

    kernel = pystencils.create_kernel(assignments).compile()
    rotated = np.zeros_like(lenna)

    kernel(x=lenna, y=rotated)

    pyconrad.imshow(lenna, "lenna")
    pyconrad.imshow(rotated, "rotated")

    # If distance in input field is twice as close we will see a smaller image
    x.coordinate_transform /= 2

    input_coordinate = x.physical_to_index(inverse_matrix @ y.physical_coordinates)

    assignments = pystencils.AssignmentCollection({
        y.center(): LinearInterpolator(x).at(input_coordinate)
    })

    kernel = pystencils.create_kernel(assignments).compile()
    rotated = np.zeros_like(lenna)

    kernel(x=lenna, y=rotated)

    pyconrad.imshow(rotated, "rotated smaller")

    # Conversely, if output field has samples 3 times closer we will see a bigger image
    y.coordinate_transform /= 3

    input_coordinate = x.physical_to_index(inverse_matrix @ y.physical_coordinates)

    assignments = pystencils.AssignmentCollection({
        y.center(): LinearInterpolator(x).at(input_coordinate)
    })

    kernel = pystencils.create_kernel(assignments).compile()
    rotated = np.zeros_like(lenna)

    kernel(x=lenna, y=rotated)

    pyconrad.imshow(rotated, "rotated bigger")

    # coordinate_transform can be any matrix, also with symbols as entries


def main():
    test_rotate_center()


if __name__ == '__main__':
    main()
