from dataclasses import dataclass

import pymbolic.primitives as pb
from pymbolic.mapper.c_code import CCodeMapper
from pymbolic.mapper.dependency import DependencyMapper

from ..typed_expressions import PsTypedVariable


@dataclass
class PsParamConstraint:
    condition: pb.Comparison
    message: str = ""

    def print_c_condition(self):
        return CCodeMapper()(self.condition)
    
    def get_variables(self) -> set[PsTypedVariable]:
        return DependencyMapper(False, False, False, False)(self.condition)
    
    def __str__(self) -> str:
        return f"{self.message} [{self.condition}]"
