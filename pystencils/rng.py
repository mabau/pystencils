import numpy as np
import sympy as sp

from pystencils import TypedSymbol
from pystencils.astnodes import LoopOverCoordinate
from pystencils.backends.cbackend import CustomCodeNode

philox_two_doubles_call = """
{result_symbols[0].dtype} {result_symbols[0].name};
{result_symbols[1].dtype} {result_symbols[1].name};
philox_double2({parameters}, {result_symbols[0].name}, {result_symbols[1].name});
"""

philox_four_floats_call = """
{result_symbols[0].dtype} {result_symbols[0].name};
{result_symbols[1].dtype} {result_symbols[1].name};
{result_symbols[2].dtype} {result_symbols[2].name};
{result_symbols[3].dtype} {result_symbols[3].name};
philox_float4({parameters},
              {result_symbols[0].name}, {result_symbols[1].name}, {result_symbols[2].name}, {result_symbols[3].name});

"""


class PhiloxTwoDoubles(CustomCodeNode):

    def __init__(self, dim, time_step=TypedSymbol("time_step", np.uint32), keys=(0, 0)):
        self.result_symbols = tuple(TypedSymbol(sp.Dummy().name, np.float64) for _ in range(2))
        symbols_read = [s for s in keys if isinstance(s, sp.Symbol)]
        super().__init__("", symbols_read=symbols_read, symbols_defined=self.result_symbols)
        self._time_step = time_step
        self.headers = ['"philox_rand.h"']
        self.keys = tuple(keys)
        self._args = sp.sympify((dim, time_step, keys))
        self._dim = dim

    @property
    def args(self):
        return self._args

    @property
    def undefined_symbols(self):
        result = {a for a in self.args if isinstance(a, sp.Symbol)}
        loop_counters = [LoopOverCoordinate.get_loop_counter_symbol(i)
                         for i in range(self._dim)]
        result.update(loop_counters)
        return result

    def fast_subs(self, _):
        return self  # nothing to replace inside this node - would destroy intermediate "dummy" by re-creating them

    def get_code(self, dialect, vector_instruction_set):
        parameters = [self._time_step] + [LoopOverCoordinate.get_loop_counter_symbol(i)
                                          for i in range(self._dim)] + list(self.keys)

        while len(parameters) < 6:
            parameters.append(0)
        parameters = parameters[:6]

        assert len(parameters) == 6

        if dialect == 'cuda' or (dialect == 'c' and vector_instruction_set is None):
            return philox_two_doubles_call.format(parameters=', '.join(str(p) for p in parameters),
                                                  result_symbols=self.result_symbols)
        else:
            raise NotImplementedError("Not yet implemented for this backend")

    def __repr__(self):
        return "{}, {} <- PhiloxRNG".format(*self.result_symbols)


class PhiloxFourFloats(CustomCodeNode):

    def __init__(self, dim, time_step=TypedSymbol("time_step", np.uint32), keys=(0, 0)):
        self.result_symbols = tuple(TypedSymbol(sp.Dummy().name, np.float32) for _ in range(4))
        symbols_read = [s for s in keys if isinstance(s, sp.Symbol)]

        super().__init__("", symbols_read=symbols_read, symbols_defined=self.result_symbols)
        self._time_step = time_step
        self.headers = ['"philox_rand.h"']
        self.keys = tuple(keys)
        self._args = sp.sympify((dim, time_step, keys))
        self._dim = dim

    @property
    def args(self):
        return self._args

    @property
    def undefined_symbols(self):
        result = {a for a in self.args if isinstance(a, sp.Symbol)}
        loop_counters = [LoopOverCoordinate.get_loop_counter_symbol(i)
                         for i in range(self._dim)]
        result.update(loop_counters)
        return result

    def fast_subs(self, _):
        return self  # nothing to replace inside this node - would destroy intermediate "dummy" by re-creating them

    def get_code(self, dialect, vector_instruction_set):
        parameters = [self._time_step] + [LoopOverCoordinate.get_loop_counter_symbol(i)
                                          for i in range(self._dim)] + list(self.keys)

        while len(parameters) < 6:
            parameters.append(0)
        parameters = parameters[:6]

        assert len(parameters) == 6

        if dialect == 'cuda' or (dialect == 'c' and vector_instruction_set is None):
            return philox_four_floats_call.format(parameters=', '.join(str(p) for p in parameters),
                                                  result_symbols=self.result_symbols)
        else:
            raise NotImplementedError("Not yet implemented for this backend")

    def __repr__(self):
        return "{}, {}, {}, {} <- PhiloxRNG".format(*self.result_symbols)


def random_symbol(assignment_list, rng_node=PhiloxTwoDoubles, *args, **kwargs):
    while True:
        node = rng_node(*args, **kwargs)
        inserted = False
        for symbol in node.result_symbols:
            if not inserted:
                assignment_list.insert(0, node)
                inserted = True
            yield symbol
