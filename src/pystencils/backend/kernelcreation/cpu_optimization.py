from __future__ import annotations

from .context import KernelCreationContext
from ..platforms import GenericCpu
from ..ast.structural import PsBlock

from ...config import CpuOptimConfig


def optimize_cpu(
    ctx: KernelCreationContext,
    platform: GenericCpu,
    kernel_ast: PsBlock,
    cfg: CpuOptimConfig,
):
    """Carry out CPU-specific optimizations according to the given configuration."""

    if cfg.loop_blocking:
        raise NotImplementedError("Loop blocking not implemented yet.")

    if cfg.vectorize is not False:
        raise NotImplementedError("Vectorization not implemented yet")

    if cfg.openmp:
        raise NotImplementedError("OpenMP not implemented yet")

    if cfg.use_cacheline_zeroing:
        raise NotImplementedError("CL-zeroing not implemented yet")
