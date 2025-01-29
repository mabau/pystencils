"""Fixtures for the pystencils test suite

This module provides a number of fixtures used by the pystencils test suite.
Use these fixtures wherever applicable to extend the code surface area covered
by your tests:

- All tests that should work for every target should use the `target` fixture
- All tests that should work with the highest optimization level for every target
  should use the `gen_config` fixture
- Use the `xp` fixture to access the correct array module (numpy or cupy) depending
  on the target

"""

import pytest

from types import ModuleType
from dataclasses import replace

import pystencils as ps

AVAILABLE_TARGETS = [ps.Target.GenericCPU]

try:
    import cupy

    AVAILABLE_TARGETS += [ps.Target.CUDA]
except ImportError:
    pass

AVAILABLE_TARGETS += ps.Target.available_vector_cpu_targets()
TARGET_IDS = [t.name for t in AVAILABLE_TARGETS]


@pytest.fixture(params=AVAILABLE_TARGETS, ids=TARGET_IDS)
def target(request) -> ps.Target:
    """Provides all code generation targets available on the current hardware"""
    return request.param


@pytest.fixture
def gen_config(target: ps.Target):
    """Default codegen configuration for the current target.

    For GPU targets, set default indexing options.
    For vector-CPU targets, set default vectorization config.
    """

    gen_config = ps.CreateKernelConfig(target=target)

    if target.is_vector_cpu():
        gen_config.cpu.vectorize.enable = True
        gen_config.cpu.vectorize.assume_inner_stride_one = True

    return gen_config


@pytest.fixture()
def xp(target: ps.Target) -> ModuleType:
    """Primary array module for the current target.

    Returns:
        `cupy` if `target == Target.CUDA`, and `numpy` otherwise
    """
    if target == ps.Target.CUDA:
        import cupy as xp

        return xp
    else:
        import numpy as np

        return np
