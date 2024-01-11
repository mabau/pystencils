import pytest

import pymbolic.primitives as pb
from pymbolic.mapper.constant_folder import ConstantFoldingMapper

from pystencils.nbackend.types.quick import *
from pystencils.nbackend.typed_expressions import PsTypedConstant


@pytest.mark.parametrize("width", (8, 16, 32, 64))
def test_constant_folding_int(width):
    folder = ConstantFoldingMapper()

    expr = pb.Sum(
        (
            PsTypedConstant(13, UInt(width)),
            PsTypedConstant(5, UInt(width)),
            PsTypedConstant(3, UInt(width)),
        )
    )

    assert folder(expr) == PsTypedConstant(21, UInt(width))

    expr = pb.Product(
        (PsTypedConstant(-1, SInt(width)), PsTypedConstant(41, SInt(width)))
    ) - PsTypedConstant(12, SInt(width))

    assert folder(expr) == PsTypedConstant(-53, SInt(width))

    expr = pb.Product(
        (
            PsTypedConstant(2, SInt(width)),
            PsTypedConstant(-3, SInt(width)),
            PsTypedConstant(4, SInt(width))
        )
    )

    assert folder(expr) == PsTypedConstant(-24, SInt(width))


@pytest.mark.parametrize("width", (32, 64))
def test_constant_folding_float(width):
    folder = ConstantFoldingMapper()

    expr = pb.Quotient(
        PsTypedConstant(14.0, Fp(width)),
        pb.Sum(
            (
                PsTypedConstant(2.5, Fp(width)),
                PsTypedConstant(4.5, Fp(width)),
            )
        ),
    )

    assert folder(expr) == PsTypedConstant(7.0, Fp(width))
