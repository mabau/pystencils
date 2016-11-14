import sympy as sp
from sympy.core.cache import cacheit


class TypedSymbol(sp.Symbol):

    def __new__(cls, name, *args, **kwds):
        obj = TypedSymbol.__xnew_cached_(cls, name, *args, **kwds)
        return obj

    def __new_stage2__(cls, name, dtype):
        obj = super(TypedSymbol, cls).__xnew__(cls, name)
        obj._dtype = dtype
        return obj

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    @property
    def dtype(self):
        return self._dtype

    def _hashable_content(self):
        superClassContents = list(super(TypedSymbol, self)._hashable_content())
        t = tuple(superClassContents + [hash(self._dtype)])
        return t
