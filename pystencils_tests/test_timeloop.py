import time
import numpy as np

from pystencils import Assignment
from pystencils import create_kernel
from pystencils.datahandling import create_data_handling
from pystencils.timeloop import TimeLoop


def test_timeloop():
    dh = create_data_handling(domain_size=(10, 10), periodicity=True)

    pre = dh.add_array('pre_run_field', values_per_cell=1)
    dh.fill("pre_run_field", 0.0, ghost_layers=True)
    f = dh.add_array('field', values_per_cell=1)
    dh.fill("field", 0.0, ghost_layers=True)
    post = dh.add_array('post_run_field', values_per_cell=1)
    dh.fill("post_run_field", 0.0, ghost_layers=True)

    pre_assignments = Assignment(pre.center, pre.center + 1)
    pre_kernel = create_kernel(pre_assignments).compile()
    assignments = Assignment(f.center, f.center + 1)
    kernel = create_kernel(assignments).compile()
    post_assignments = Assignment(post.center, post.center + 1)
    post_kernel = create_kernel(post_assignments).compile()

    timeloop = TimeLoop(steps=1)

    def pre_run():
        dh.run_kernel(pre_kernel)

    def post_run():
        dh.run_kernel(post_kernel)

    timeloop.add_pre_run_function(pre_run)
    timeloop.add_post_run_function(post_run)
    timeloop.add_call(kernel, {'field': dh.cpu_arrays["field"]})

    timeloop.run(time_steps=5)
    assert np.all(dh.cpu_arrays["pre_run_field"] == 1.0)
    assert np.all(dh.cpu_arrays["field"] == 5.0)
    assert np.all(dh.cpu_arrays["post_run_field"] == 1.0)

    seconds = 2
    start = time.perf_counter()
    timeloop.run_time_span(seconds=seconds)
    end = time.perf_counter()

    np.testing.assert_almost_equal(seconds, end - start, decimal=3)
