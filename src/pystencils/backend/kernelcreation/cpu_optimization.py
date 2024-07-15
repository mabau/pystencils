from __future__ import annotations
from typing import cast, TYPE_CHECKING

from .context import KernelCreationContext
from ..ast.structural import PsBlock

from ...config import CpuOptimConfig, OpenMpConfig

if TYPE_CHECKING:
    from ..platforms import GenericCpu


def optimize_cpu(
    ctx: KernelCreationContext,
    platform: GenericCpu,
    kernel_ast: PsBlock,
    cfg: CpuOptimConfig | None,
) -> PsBlock:
    """Carry out CPU-specific optimizations according to the given configuration."""
    from ..transformations import CanonicalizeSymbols, HoistLoopInvariantDeclarations

    canonicalize = CanonicalizeSymbols(ctx, True)
    kernel_ast = cast(PsBlock, canonicalize(kernel_ast))

    hoist_invariants = HoistLoopInvariantDeclarations(ctx)
    kernel_ast = cast(PsBlock, hoist_invariants(kernel_ast))

    if cfg is None:
        return kernel_ast

    if cfg.loop_blocking:
        raise NotImplementedError("Loop blocking not implemented yet.")

    if cfg.vectorize is not False:
        raise NotImplementedError("Vectorization not implemented yet")

    if cfg.openmp is not False:
        from ..transformations import AddOpenMP

        params = cfg.openmp if isinstance(cfg.openmp, OpenMpConfig) else OpenMpConfig()
        add_omp = AddOpenMP(ctx, params)
        kernel_ast = cast(PsBlock, add_omp(kernel_ast))

    if cfg.use_cacheline_zeroing:
        raise NotImplementedError("CL-zeroing not implemented yet")

    return kernel_ast
