import numpy as np
import waLBerla as wlb

import pystencils
from pystencils import make_slice

from pathlib import Path

from pystencils.boundaries import BoundaryHandling, Neumann
from pystencils.slicing import slice_from_direction

from pystencils.datahandling.parallel_datahandling import ParallelDataHandling
from pystencils.datahandling import create_data_handling
from pystencils_tests.test_datahandling import (
    access_and_gather, kernel_execution_jacobi, reduction, synchronization, vtk_output)

SCRIPT_FOLDER = Path(__file__).parent.absolute()
INPUT_FOLDER = SCRIPT_FOLDER / "test_data"

try:
    import pytest
except ImportError:
    import unittest.mock
    pytest = unittest.mock.MagicMock()


def test_access_and_gather():
    block_size = (4, 7, 1)
    num_blocks = (3, 2, 1)
    cells = tuple(a * b for a, b in zip(block_size, num_blocks))
    blocks = wlb.createUniformBlockGrid(blocks=num_blocks, cellsPerBlock=block_size, oneBlockPerProcess=False,
                                        periodic=(1, 1, 1))
    dh = ParallelDataHandling(blocks, default_ghost_layers=2)
    access_and_gather(dh, cells)
    synchronization(dh, test_gpu=False)
    if hasattr(wlb, 'cuda'):
        synchronization(dh, test_gpu=True)


def test_gpu():
    pytest.importorskip('waLBerla.cuda')

    block_size = (4, 7, 1)
    num_blocks = (3, 2, 1)
    blocks = wlb.createUniformBlockGrid(blocks=num_blocks, cellsPerBlock=block_size, oneBlockPerProcess=False)
    dh = ParallelDataHandling(blocks, default_ghost_layers=2)
    dh.add_array('v', values_per_cell=3, dtype=np.int64, ghost_layers=2, gpu=True)

    for b in dh.iterate():
        b['v'].fill(42)
    dh.all_to_gpu()
    for b in dh.iterate():
        b['v'].fill(0)
    dh.to_cpu('v')
    for b in dh.iterate():
        np.testing.assert_equal(b['v'], 42)


@pytest.mark.parametrize('target', (pystencils.Target.CPU, pystencils.Target.GPU))
def test_kernel(target):
    if target == pystencils.Target.GPU:
        pytest.importorskip('waLBerla.cuda')

    # 3D
    blocks = wlb.createUniformBlockGrid(blocks=(3, 2, 4), cellsPerBlock=(3, 2, 5), oneBlockPerProcess=False)
    dh = ParallelDataHandling(blocks, default_target=target)
    kernel_execution_jacobi(dh, target)
    reduction(dh)

    # 2D
    blocks = wlb.createUniformBlockGrid(blocks=(3, 2, 1), cellsPerBlock=(3, 2, 1), oneBlockPerProcess=False)
    dh = ParallelDataHandling(blocks, dim=2, default_target=target)
    kernel_execution_jacobi(dh, target)
    reduction(dh)


def test_vtk_output():
    blocks = wlb.createUniformBlockGrid(blocks=(3, 2, 4), cellsPerBlock=(3, 2, 5), oneBlockPerProcess=False)
    dh = ParallelDataHandling(blocks)
    vtk_output(dh)


def test_block_iteration():
    block_size = (16, 16, 16)
    num_blocks = (2, 2, 2)
    blocks = wlb.createUniformBlockGrid(blocks=num_blocks, cellsPerBlock=block_size, oneBlockPerProcess=False)
    dh = ParallelDataHandling(blocks, default_ghost_layers=2)
    dh.add_array('v', values_per_cell=1, dtype=np.int64, ghost_layers=2)

    for b in dh.iterate():
        b['v'].fill(1)

    s = 0
    for b in dh.iterate():
        s += np.sum(b['v'])

    assert s == 40*40*40

    sl = make_slice[0:18, 0:18, 0:18]
    for b in dh.iterate(slice_obj=sl):
        b['v'].fill(0)

    s = 0
    for b in dh.iterate():
        s += np.sum(b['v'])

    assert s == 40*40*40 - 20*20*20


def test_getter_setter():
    pytest.importorskip('waLBerla.cuda')

    block_size = (2, 2, 2)
    num_blocks = (2, 2, 2)
    blocks = wlb.createUniformBlockGrid(blocks=num_blocks, cellsPerBlock=block_size, oneBlockPerProcess=False)
    dh = ParallelDataHandling(blocks, default_ghost_layers=2, default_target=pystencils.Target.GPU)
    dh.add_array('v', values_per_cell=1, dtype=np.int64, ghost_layers=2, gpu=True)

    assert dh.shape == (4, 4, 4)
    assert dh.periodicity == (False, False, False)
    assert dh.values_per_cell('v') == 1
    assert dh.has_data('v') is True
    assert 'v' in dh.array_names
    dh.log_on_root()
    assert dh.is_root is True
    assert dh.world_rank == 0

    dh.to_gpu('v')
    assert dh.is_on_gpu('v') is True
    dh.all_to_cpu()


def test_parallel_datahandling_boundary_conditions():
    pytest.importorskip('waLBerla.cuda')

    dh = create_data_handling(domain_size=(7, 7), periodicity=True, parallel=True,
                              default_target=pystencils.Target.GPU)

    src = dh.add_array('src', values_per_cell=1)
    dh.fill(src.name, 0.0, ghost_layers=True)
    dh.fill(src.name, 1.0, ghost_layers=False)

    src2 = dh.add_array('src2', values_per_cell=1)

    src_cpu = dh.add_array('src_cpu', values_per_cell=1, gpu=False)
    dh.fill(src_cpu.name, 0.0, ghost_layers=True)
    dh.fill(src_cpu.name, 1.0, ghost_layers=False)

    boundary_stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    boundary_handling_cpu = BoundaryHandling(dh, src_cpu.name, boundary_stencil,
                                             name="boundary_handling_cpu", target=pystencils.Target.CPU)

    boundary_handling = BoundaryHandling(dh, src.name, boundary_stencil,
                                         name="boundary_handling_gpu", target=pystencils.Target.GPU)

    neumann = Neumann()
    for d in ('N', 'S', 'W', 'E'):
        boundary_handling.set_boundary(neumann, slice_from_direction(d, dim=2))
        boundary_handling_cpu.set_boundary(neumann, slice_from_direction(d, dim=2))

    boundary_handling.prepare()
    boundary_handling_cpu.prepare()

    boundary_handling_cpu()

    dh.all_to_gpu()
    boundary_handling()
    dh.all_to_cpu()
    for block in dh.iterate():
        np.testing.assert_almost_equal(block[src_cpu.name], block[src.name])

    assert dh.custom_data_names == ('boundary_handling_cpuIndexArrays', 'boundary_handling_gpuIndexArrays')
    dh.swap(src.name, src2.name, gpu=True)


def test_save_data():
    domain_shape = (2, 2)

    dh = create_data_handling(domain_size=domain_shape, default_ghost_layers=1, parallel=True)
    dh.add_array("src", values_per_cell=9)
    dh.fill("src", 1.0, ghost_layers=True)
    dh.add_array("dst", values_per_cell=9)
    dh.fill("dst", 1.0, ghost_layers=True)

    dh.save_all(str(INPUT_FOLDER) + '/datahandling_parallel_save_test')


def test_load_data():
    domain_shape = (2, 2)

    dh = create_data_handling(domain_size=domain_shape, default_ghost_layers=1, parallel=True)
    dh.add_array("src", values_per_cell=9)
    dh.fill("src", 0.0, ghost_layers=True)
    dh.add_array("dst", values_per_cell=9)
    dh.fill("dst", 0.0, ghost_layers=True)

    dh.load_all(str(INPUT_FOLDER) + '/datahandling_parallel_load_test')
    assert np.all(dh.gather_array('src')) == 1
    assert np.all(dh.gather_array('src')) == 1
