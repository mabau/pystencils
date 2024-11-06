import numpy as np
import pytest

import pystencils as ps
from pystencils.assignment import assignment_from_stencil


def test_assignment_from_stencil():

    stencil = [
        [0, 0, 4, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0]
    ]

    x, y = ps.fields('x, y: [2D]')

    assignment = assignment_from_stencil(stencil, x, y)
    assert isinstance(assignment, ps.Assignment)
    assert assignment.rhs == x[0, 1] + 4 * x[-1, 1] + 2 * x[0, 0] + 3 * x[0, -1]

    assignment = assignment_from_stencil(stencil, x, y, normalization_factor=1 / np.sum(stencil))
    assert isinstance(assignment, ps.Assignment)



@pytest.mark.parametrize('target', [ps.Target.CPU, ps.Target.GPU])
def test_add_augmented_assignment(target):
    if target == ps.Target.GPU:
        pytest.importorskip("cupy")

    domain_size = (5, 5)
    dh = ps.create_data_handling(domain_size=domain_size, periodicity=True, default_target=target)

    f = dh.add_array("f", values_per_cell=1)
    dh.fill(f.name, 0.0)

    g = dh.add_array("g", values_per_cell=1)
    dh.fill(g.name, 1.0)

    up = ps.AddAugmentedAssignment(f.center, g.center)

    config = ps.CreateKernelConfig(target=dh.default_target)
    ast = ps.create_kernel(up, config=config)

    kernel = ast.compile()
    for i in range(10):
        dh.run_kernel(kernel)

    if target == ps.Target.GPU:
        dh.all_to_cpu()

    result = dh.gather_array(f.name)

    for x in range(domain_size[0]):
        for y in range(domain_size[1]):
            assert result[x, y] == 10
