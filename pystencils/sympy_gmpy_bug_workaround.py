# Disable gmpy backend until this bug is resolved if joblib serialize
# See https://github.com/sympy/sympy/pull/13530
import os
import warnings

os.environ['MPMATH_NOGMPY'] = '1'
try:
    import mpmath.libmp
    # In case the user has imported sympy first, then pystencils
    if mpmath.libmp.BACKEND == 'gmpy':
        warnings.warn("You are using the gmpy backend. You might encounter an error 'argument is not an mpz sympy'. "
                      "This is due to a known bug in sympy/gmpy library. "
                      "To prevent this, import pystencils first then sympy or set the environment variable "
                      "MPMATH_NOGMPY=1")
except ImportError:
    pass

__all__ = []
