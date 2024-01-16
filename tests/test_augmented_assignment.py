import pytest
import pystencils as ps


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
