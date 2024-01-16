import pystencils as ps

from pystencils.astnodes import Block, Conditional, SympyAssignment


def test_dot_print():
    src, dst = ps.fields("src, dst: double[2D]", layout='c')

    true_block = Block([SympyAssignment(dst[0, 0], src[-1, 0])])
    false_block = Block([SympyAssignment(dst[0, 0], src[1, 0])])
    ur = [true_block, Conditional(dst.center() > 0.0, true_block, false_block)]

    ast = ps.create_kernel(ur)
