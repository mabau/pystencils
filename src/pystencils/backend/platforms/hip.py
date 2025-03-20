from __future__ import annotations

from .generic_gpu import GenericGpu


class HipPlatform(GenericGpu):
    """Platform for the HIP GPU taret."""

    @property
    def required_headers(self) -> set[str]:
        return {'"pystencils_runtime/hip.h"'}
