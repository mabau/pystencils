import numpy as np
import pytest

import pystencils


def test_dtype_check_wrong_type():
    array = np.ones((10, 20)).astype(np.float32)
    output = np.zeros_like(array)
    x, y = pystencils.fields('x,y: [2D]')
    stencil = [[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]]
    assignment = pystencils.assignment.assignment_from_stencil(stencil, x, y, normalization_factor=1 / np.sum(stencil))
    kernel = pystencils.create_kernel([assignment]).compile()

    with pytest.raises(ValueError) as e:
        kernel(x=array, y=output)
    assert 'Wrong data type' in str(e.value)


def test_dtype_check_correct_type():
    array = np.ones((10, 20)).astype(np.float64)
    output = np.zeros_like(array)
    x, y = pystencils.fields('x,y: [2D]')
    stencil = [[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]]
    assignment = pystencils.assignment.assignment_from_stencil(stencil, x, y, normalization_factor=1 / np.sum(stencil))
    kernel = pystencils.create_kernel([assignment]).compile()
    kernel(x=array, y=output)
    assert np.allclose(output[1:-1, 1:-1], np.ones_like(output[1:-1, 1:-1]))
