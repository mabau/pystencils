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


def _get_rng_code(template, dialect, vector_instruction_set, args, result_symbols):
    if dialect == 'cuda' or (dialect == 'c' and vector_instruction_set is None):
        return template.format(parameters=', '.join(str(a) for a in args),
                               result_symbols=result_symbols)
    else:
        raise NotImplementedError("Not yet implemented for this backend")


class RNGBase(CustomCodeNode):

    id = 0

    def __init__(self, dim, time_step=TypedSymbol("time_step", np.uint32), offsets=None, keys=None):
        if keys is None:
            keys = (0,) * self._num_keys
        if offsets is None:
            offsets = (0,) * dim
        if len(keys) != self._num_keys:
            raise ValueError(f"Provided {len(keys)} keys but need {self._num_keys}")
        if len(offsets) != dim:
            raise ValueError(f"Provided {len(offsets)} offsets but need {dim}")
        coordinates = [LoopOverCoordinate.get_loop_counter_symbol(i) + offsets[i] for i in range(dim)]
        if dim < 3:
            coordinates.append(0)

        self._args = sp.sympify([time_step, *coordinates, *keys])
        self.result_symbols = tuple(TypedSymbol(f'random_{self.id}_{i}', self._data_type)
                                    for i in range(self._num_vars))
        symbols_read = set.union(*[s.atoms(sp.Symbol) for s in self.args])
        super().__init__("", symbols_read=symbols_read, symbols_defined=self.result_symbols)

        self.headers = [f'"{self._name}_rand.h"']

        RNGBase.id += 1

    @property
    def args(self):
        return self._args

    def get_code(self, dialect, vector_instruction_set):
        template = _get_rng_template(self._name, self._data_type, self._num_vars)
        return _get_rng_code(template, dialect, vector_instruction_set, self.args, self.result_symbols)

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
