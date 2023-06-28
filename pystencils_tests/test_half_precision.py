import pytest
import platform

import numpy as np
import pystencils as ps


@pytest.mark.parametrize('target', (ps.Target.CPU, ps.Target.GPU))
def test_half_precison(target):
    if target == ps.Target.CPU:
        if not platform.machine() in ['arm64', 'aarch64']:
            pytest.xfail("skipping half precision test on non arm platform")

        if 'clang' not in ps.cpu.cpujit.get_compiler_config()['command']:
            pytest.xfail("skipping half precision because clang compiler is not used")

    if target == ps.Target.GPU:
        pytest.importorskip("cupy")

    dh = ps.create_data_handling(domain_size=(10, 10), default_target=target)

    f1 = dh.add_array("f1", values_per_cell=1, dtype=np.float16)
    dh.fill("f1", 1.0, ghost_layers=True)
    f2 = dh.add_array("f2", values_per_cell=1, dtype=np.float16)
    dh.fill("f2", 2.0, ghost_layers=True)

    f3 = dh.add_array("f3", values_per_cell=1, dtype=np.float16)
    dh.fill("f3", 0.0, ghost_layers=True)

    up = ps.Assignment(f3.center, f1.center + 2.1 * f2.center)

    config = ps.CreateKernelConfig(target=dh.default_target, default_number_float=np.float32)
    ast = ps.create_kernel(up, config=config)

    kernel = ast.compile()

    dh.run_kernel(kernel)
    dh.all_to_cpu()

    assert np.all(dh.cpu_arrays[f3.name] == 5.2)
    assert dh.cpu_arrays[f3.name].dtype == np.float16
