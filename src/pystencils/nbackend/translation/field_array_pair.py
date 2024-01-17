from dataclasses import dataclass

from ...field import Field
from ..arrays import PsLinearizedArray, PsArrayBasePointer
from ..types import PsIntegerType
from ..constraints import PsKernelConstraint

from .iteration_domain import PsIterationDomain

@dataclass
class PsFieldArrayPair:
    field: Field
    array: PsLinearizedArray
    base_ptr: PsArrayBasePointer


@dataclass
class PsDomainFieldArrayPair(PsFieldArrayPair):
    ghost_layers: int
    interior_base_ptr: PsArrayBasePointer
    domain: PsIterationDomain
