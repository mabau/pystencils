import pytest

import sympy as sp
import pystencils as ps
from pystencils.astnodes import LoopOverCoordinate, Conditional, Block, SympyAssignment


@pytest.mark.parametrize('target', [ps.Target.CPU, ps.Target.GPU])
@pytest.mark.parametrize('iteration_slice', [False, True])
def test_mod(target, iteration_slice):
    if target == ps.Target.GPU:
        pytest.importorskip("cupy")
    dh = ps.create_data_handling(domain_size=(5, 5), periodicity=True, default_target=ps.Target.CPU)

    loop_ctrs = [LoopOverCoordinate.get_loop_counter_symbol(i) for i in range(dh.dim)]
    cond = [sp.Eq(sp.Mod(loop_ctrs[i], 2), 1) for i in range(dh.dim)]

    field = dh.add_array("a", values_per_cell=1)

    eq_list = [SympyAssignment(field.center, 1.0)]

    if iteration_slice:
        iteration_slice = ps.make_slice[1:-1:2, 1:-1:2]
        config = ps.CreateKernelConfig(target=dh.default_target, iteration_slice=iteration_slice)
        assign = eq_list
    else:
        assign = [Conditional(sp.And(*cond), Block(eq_list))]
        config = ps.CreateKernelConfig(target=dh.default_target)

    kernel = ps.create_kernel(assign, config=config).compile()

    dh.fill(field.name, 0, ghost_layers=True)

    if config.target == ps.enums.Target.GPU:
        dh.to_gpu(field.name)

    dh.run_kernel(kernel)

    if config.target == ps.enums.Target.GPU:
        dh.to_cpu(field.name)

    result = dh.gather_array(field.name, ghost_layers=True)

    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            if x % 2 == 1 and y % 2 == 1:
                assert result[x, y] == 1.0
            else:
                assert result[x, y] == 0.0
