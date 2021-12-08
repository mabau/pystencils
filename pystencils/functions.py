import sympy as sp


class DivFunc(sp.Function):
    # TODO: documentation
    is_Atom = True
    is_real = True

    def __new__(cls, *args, **kwargs):
        if len(args) != 2:
            raise ValueError(f'{cls} takes only 2 arguments, instead {len(args)} received!')
        divisor, dividend, *other_args = args

        return sp.Function.__new__(cls, divisor, dividend, *other_args, **kwargs)

    def _eval_evalf(self, *args, **kwargs):
        return self.divisor.evalf() / self.dividend.evalf()

    @property
    def divisor(self):
        return self.args[0]

    @property
    def dividend(self):
        return self.args[1]

