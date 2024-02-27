#   TODO: Reimplement for constant folder
# import pytest

# from pystencils.backend.types.quick import *
# from pystencils.backend.constants import PsConstant


# @pytest.mark.parametrize("width", (8, 16, 32, 64))
# def test_constant_folding_int(width):
#     folder = ConstantFoldingMapper()

#     expr = pb.Sum(
#         (
#             PsTypedConstant(13, UInt(width)),
#             PsTypedConstant(5, UInt(width)),
#             PsTypedConstant(3, UInt(width)),
#         )
#     )

#     assert folder(expr) == PsTypedConstant(21, UInt(width))

#     expr = pb.Product(
#         (PsTypedConstant(-1, SInt(width)), PsTypedConstant(41, SInt(width)))
#     ) - PsTypedConstant(12, SInt(width))

#     assert folder(expr) == PsTypedConstant(-53, SInt(width))
