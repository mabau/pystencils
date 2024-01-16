from dataclasses import dataclass

import pymbolic.primitives as pb
from pymbolic.mapper.c_code import CCodeMapper


@dataclass
class PsParamConstraint:
    condition: pb.Comparison
    message: str = ""

    def print(self):
        return CCodeMapper()(self.condition)
