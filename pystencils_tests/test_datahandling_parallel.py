import numpy as np
import waLBerla as wlb
from pystencils import make_slice

from pystencils.datahandling.parallel_datahandling import ParallelDataHandling
from pystencils_tests.test_datahandling import (
    access_and_gather, kernel_execution_jacobi, reduction, synchronization, vtk_output)

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
    if not hasattr(wlb, 'cuda'):
        print("Skip GPU tests because walberla was built without CUDA")
        return

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


def test_kernel():

    for gpu in (True, False):
        if gpu and not hasattr(wlb, 'cuda'):
            print("Skipping CUDA tests because walberla was built without GPU support")
            continue

        # 3D
        blocks = wlb.createUniformBlockGrid(blocks=(3, 2, 4), cellsPerBlock=(3, 2, 5), oneBlockPerProcess=False)
        dh = ParallelDataHandling(blocks)
        kernel_execution_jacobi(dh, 'gpu')
        reduction(dh)

        # 2D
        blocks = wlb.createUniformBlockGrid(blocks=(3, 2, 1), cellsPerBlock=(3, 2, 1), oneBlockPerProcess=False)
        dh = ParallelDataHandling(blocks, dim=2)
        kernel_execution_jacobi(dh, 'gpu')
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
    dh.add_array('v', values_per_cell=1, dtype=np.int64, ghost_layers=2, gpu=True)

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
    block_size = (2, 2, 2)
    num_blocks = (2, 2, 2)
    blocks = wlb.createUniformBlockGrid(blocks=num_blocks, cellsPerBlock=block_size, oneBlockPerProcess=False)
    dh = ParallelDataHandling(blocks, default_ghost_layers=2)
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
