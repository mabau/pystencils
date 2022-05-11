import copy
import numpy as np
import sympy as sp

from pystencils.typing import TypedSymbol, CastFunc
from pystencils.astnodes import LoopOverCoordinate
from pystencils.backends.cbackend import CustomCodeNode
from pystencils.sympyextensions import fast_subs


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

        self.headers = [f'"{self._name.split("_")[0]}_rand.h"']

        RNGBase.id += 1

    @property
    def args(self):
        return self._args

    def fast_subs(self, subs_dict, skip):
        rng = copy.deepcopy(self)
        rng._args = [fast_subs(a, subs_dict, skip) for a in rng._args]
        return rng

    def get_code(self, dialect, vector_instruction_set, print_arg):
        code = "\n"
        for r in self.result_symbols:
            if vector_instruction_set and not self.args[1].atoms(CastFunc):
                # this vector RNG has become scalar through substitution
                code += f"{r.dtype} {r.name};\n"
            else:
                code += f"{vector_instruction_set[r.dtype.c_name] if vector_instruction_set else r.dtype} " + \
                        f"{r.name};\n"
        args = [print_arg(a) for a in self.args] + ['' + r.name for r in self.result_symbols]
        code += (self._name + "(" + ", ".join(args) + ");\n")
        return code

    def __repr__(self):
        return ", ".join([str(s) for s in self.result_symbols]) + " \\leftarrow " + \
            self._name.capitalize() + "_RNG(" + ", ".join([str(a) for a in self.args]) + ")"


class PhiloxTwoDoubles(RNGBase):
    _name = "philox_double2"
    _data_type = np.float64
    _num_vars = 2
    _num_keys = 2


class PhiloxFourFloats(RNGBase):
    _name = "philox_float4"
    _data_type = np.float32
    _num_vars = 4
    _num_keys = 2


class AESNITwoDoubles(RNGBase):
    _name = "aesni_double2"
    _data_type = np.float64
    _num_vars = 2
    _num_keys = 4


class AESNIFourFloats(RNGBase):
    _name = "aesni_float4"
    _data_type = np.float32
    _num_vars = 4
    _num_keys = 4


def random_symbol(assignment_list, dim, seed=TypedSymbol("seed", np.uint32), rng_node=PhiloxTwoDoubles,
                  time_step=TypedSymbol("time_step", np.uint32), offsets=None):
    """Return a symbol generator for random numbers
    
    Args:
        assignment_list: the subexpressions member of an AssignmentCollection, into which helper variables assignments
                         will be inserted
        dim: 2 or 3 for two or three spatial dimensions
        seed: an integer or TypedSymbol(..., np.uint32) to seed the random number generator. If you create multiple
              symbol generators, please pass them different seeds so you don't get the same stream of random numbers!
        rng_node: which random number generator to use (PhiloxTwoDoubles, PhiloxFourFloats, AESNITwoDoubles,
                  AESNIFourFloats).
        time_step: TypedSymbol(..., np.uint32) that indicates the number of the current time step
        offsets: tuple of offsets (constant integers or TypedSymbol(..., np.uint32)) that give the global coordinates
                 of the local origin
    """
    counter = 0
    while True:
        keys = (counter, seed) + (0,) * (rng_node._num_keys - 2)
        node = rng_node(dim, keys=keys, time_step=time_step, offsets=offsets)
        inserted = False
        for symbol in node.result_symbols:
            if not inserted:
                assignment_list.insert(0, node)
                inserted = True
            yield symbol
        counter += 1
