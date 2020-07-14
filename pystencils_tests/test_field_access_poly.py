#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""


from pystencils.session import *

from sympy import poly


def test_field_access_poly():
    dh = ps.create_data_handling((20, 20))
    ρ = dh.add_array('rho')
    rho = ρ.center
    a = poly(rho+0.5, rho)
    print(a)


def test_field_access_piecewise():
    dh = ps.create_data_handling((20, 20))
    ρ = dh.add_array('rho')
    pw = sp.Piecewise((0, 1 < sp.Max(-0.5, ρ.center+0.5)), (1, True))
    a = sp.simplify(pw)
    print(a)
