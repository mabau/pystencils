import numpy as np

import pystencils
from pystencils.sympyextensions.astnodes import assignment_from_stencil


def test_assignment_from_stencil():

    stencil = [
        [0, 0, 4, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0]
    ]

    x, y = pystencils.fields('x, y: [2D]')

    assignment = assignment_from_stencil(stencil, x, y)
    assert isinstance(assignment, pystencils.Assignment)
    assert assignment.rhs == x[0, 1] + 4 * x[-1, 1] + 2 * x[0, 0] + 3 * x[0, -1]

    assignment = assignment_from_stencil(stencil, x, y, normalization_factor=1 / np.sum(stencil))
    assert isinstance(assignment, pystencils.Assignment)
