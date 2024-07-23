from .sympyextensions.bit_masks import flag_cond as _flag_cond

from warnings import warn
warn(
    "Importing the `pystencils.bit_masks` module is deprecated. "
    "Import `flag_cond` from `pystencils.sympyextensions` instead."
)

flag_cond = _flag_cond
