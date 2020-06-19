import numpy as np
import sympy as sp

from pystencils import TypedSymbol
from pystencils.astnodes import LoopOverCoordinate
from pystencils.backends.cbackend import CustomCodeNode


def _get_rng_template(name, data_type, num_vars):
    if data_type is np.float32:
        c_type = "float"
    elif data_type is np.float64:
        c_type = "double"
    template = "\n"
    for i in range(num_vars):
        template += f"{{result_symbols[{i}].dtype}} {{result_symbols[{i}].name}};\n"
    template += ("{}_{}{}({{parameters}}, " + ", ".join(["{{result_symbols[{}].name}}"] * num_vars) + ");\n") \
        .format(name, c_type, num_vars, *tuple(range(num_vars)))
    return template


def _get_rng_code(template, dialect, vector_instruction_set, time_step, offsets, keys, dim, result_symbols):
    parameters = [time_step] + [LoopOverCoordinate.get_loop_counter_symbol(i) + offsets[i]
                                for i in range(dim)] + [0] * (3 - dim) + list(keys)

    if dialect == 'cuda' or (dialect == 'c' and vector_instruction_set is None):
        return template.format(parameters=', '.join(str(p) for p in parameters),
                               result_symbols=result_symbols)
    else:
        raise NotImplementedError("Not yet implemented for this backend")


class RNGBase(CustomCodeNode):

    def __init__(self, dim, time_step=TypedSymbol("time_step", np.uint32), offsets=(0, 0, 0), keys=None):
        if keys is None:
            keys = (0,) * self._num_keys
        if len(keys) != self._num_keys:
            raise ValueError(f"Provided {len(keys)} keys but need {self._num_keys}")
        if len(offsets) != 3:
            raise ValueError(f"Provided {len(offsets)} offsets but need {3}")
        self.result_symbols = tuple(TypedSymbol(sp.Dummy().name, self._data_type) for _ in range(self._num_vars))
        symbols_read = [s for s in keys if isinstance(s, sp.Symbol)]
        super().__init__("", symbols_read=symbols_read, symbols_defined=self.result_symbols)
        self._time_step = time_step
        self._offsets = offsets
        self.headers = [f'"{self._name}_rand.h"']
        self.keys = tuple(keys)
        self._args = sp.sympify((dim, time_step, keys))
        self._dim = dim

    @property
    def args(self):
        return self._args

    @property
    def undefined_symbols(self):
        result = {a for a in (self._time_step, *self._offsets, *self.keys) if isinstance(a, sp.Symbol)}
        loop_counters = [LoopOverCoordinate.get_loop_counter_symbol(i)
                         for i in range(self._dim)]
        result.update(loop_counters)
        return result

    def fast_subs(self, *_):
        return self  # nothing to replace inside this node - would destroy intermediate "dummy" by re-creating them

    def get_code(self, dialect, vector_instruction_set):
        template = _get_rng_template(self._name, self._data_type, self._num_vars)
        return _get_rng_code(template, dialect, vector_instruction_set,
                             self._time_step, self._offsets, self.keys, self._dim, self.result_symbols)

    def __repr__(self):
        return (", ".join(['{}'] * self._num_vars) + " \\leftarrow {}RNG").format(*self.result_symbols,
                                                                                  self._name.capitalize())


class PhiloxTwoDoubles(RNGBase):
    _name = "philox"
    _data_type = np.float64
    _num_vars = 2
    _num_keys = 2


class PhiloxFourFloats(RNGBase):
    _name = "philox"
    _data_type = np.float32
    _num_vars = 4
    _num_keys = 2


class AESNITwoDoubles(RNGBase):
    _name = "aesni"
    _data_type = np.float64
    _num_vars = 2
    _num_keys = 4


class AESNIFourFloats(RNGBase):
    _name = "aesni"
    _data_type = np.float32
    _num_vars = 4
    _num_keys = 4


def random_symbol(assignment_list, seed=TypedSymbol("seed", np.uint32), rng_node=PhiloxTwoDoubles, *args, **kwargs):
    counter = 0
    while True:
        node = rng_node(*args, keys=(counter, seed), **kwargs)
        inserted = False
        for symbol in node.result_symbols:
            if not inserted:
                assignment_list.insert(0, node)
                inserted = True
            yield symbol
        counter += 1
