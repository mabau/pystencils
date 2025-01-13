from .codegen import Target
from .codegen import create_kernel as _create_kernel

from warnings import warn

warn(
    "Importing anything from `pystencils.kernelcreation` is deprecated "
    "and the module will be removed in pystencils 2.1. "
    "Import from `pystencils` instead.",
    FutureWarning,
)


create_kernel = _create_kernel


def create_staggered_kernel(
    assignments, target: Target = Target.CPU, gpu_exclusive_conditions=False, **kwargs
):
    raise NotImplementedError(
        "Staggered kernels are not yet implemented for pystencils 2.0"
    )
