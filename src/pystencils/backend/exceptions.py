"""Errors and Exceptions raised by the backend during kernel translation."""


class PsInternalCompilerError(Exception):
    """Indicates an internal error during kernel translation, most likely due to a bug inside pystencils."""


class PsOptionsError(Exception):
    """Indicates an option clash in the `CreateKernelConfig`."""


class PsInputError(Exception):
    """Indicates unsupported user input to the translation system"""


class KernelConstraintsError(Exception):
    """Indicates a constraint violation in the symbolic kernel"""


class MaterializationError(Exception):
    """Indicates a fatal error during materialization of any abstract kernel component."""
