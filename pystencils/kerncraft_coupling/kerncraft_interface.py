import warnings
from collections import defaultdict
from tempfile import TemporaryDirectory
from typing import Optional

import kerncraft
import sympy as sp
from kerncraft.kerncraft import KernelCode
from kerncraft.machinemodel import MachineModel

from pystencils.astnodes import (
    KernelFunction, LoopOverCoordinate, ResolvedFieldAccess, SympyAssignment)
from pystencils.field import get_layout_from_strides
from pystencils.kerncraft_coupling.generate_benchmark import generate_benchmark
from pystencils.sympyextensions import count_operations_in_ast
from pystencils.transformations import filtered_tree_iteration
from pystencils.utils import DotDict


class PyStencilsKerncraftKernel(KernelCode):
    """
    Implementation of kerncraft's kernel interface for pystencils CPU kernels.
    Analyses a list of equations assuming they will be executed on a CPU
    """
    LIKWID_BASE = '/usr/local/likwid'

    def __init__(self, ast: KernelFunction, machine: Optional[MachineModel] = None,
                 assumed_layout='SoA', debug_print=False, filename=None):
        """Create a kerncraft kernel using a pystencils AST

        Args:
            ast: pystencils ast
            machine: kerncraft machine model - specify this if kernel needs to be compiled
            assumed_layout: either 'SoA' or 'AoS' - if fields have symbolic sizes the layout of the index
                    coordinates is not known. In this case either a structures of array (SoA) or
                    array of structures (AoS) layout is assumed
        """
        kerncraft.kernel.Kernel.__init__(self, machine)

        # Initialize state
        self.asm_block = None
        self._filename = filename

        self.kernel_ast = ast
        self.temporary_dir = TemporaryDirectory()
        self._keep_intermediates = debug_print

        # Loops
        inner_loops = [l for l in filtered_tree_iteration(ast, LoopOverCoordinate, stop_type=SympyAssignment)
                       if l.is_innermost_loop]
        if len(inner_loops) == 0:
            raise ValueError("No loop found in pystencils AST")
        else:
            if len(inner_loops) > 1:
                warnings.warn("pystencils AST contains multiple inner loops. "
                              "Only one can be analyzed - choosing first one")
            inner_loop = inner_loops[0]

        self._loop_stack = []
        cur_node = inner_loop
        while cur_node is not None:
            if isinstance(cur_node, LoopOverCoordinate):
                loop_counter_sym = cur_node.loop_counter_symbol
                loop_info = (loop_counter_sym.name, cur_node.start, cur_node.stop, 1)
                # If the correct step were to be provided, all access within that step length will
                # also need to be passed to kerncraft: cur_node.step)
                self._loop_stack.append(loop_info)
            cur_node = cur_node.parent
        self._loop_stack = list(reversed(self._loop_stack))

        # Data sources & destinations
        self.sources = defaultdict(list)
        self.destinations = defaultdict(list)

        def get_layout_tuple(f):
            if f.has_fixed_shape:
                return get_layout_from_strides(f.strides)
            else:
                layout_list = list(f.layout)
                for _ in range(f.index_dimensions):
                    layout_list.insert(0 if assumed_layout == 'SoA' else -1, max(layout_list) + 1)
                return layout_list

        reads, writes = search_resolved_field_accesses_in_ast(inner_loop)
        for accesses, target_dict in [(reads, self.sources), (writes, self.destinations)]:
            for fa in accesses:
                coord = [sp.Symbol(LoopOverCoordinate.get_loop_counter_name(i), positive=True, integer=True) + off
                         for i, off in enumerate(fa.offsets)]
                coord += list(fa.idx_coordinate_values)
                layout = get_layout_tuple(fa.field)
                permuted_coord = [sp.sympify(coord[i]) for i in layout]
                target_dict[fa.field.name].append(permuted_coord)

        # Variables (arrays)
        fields_accessed = ast.fields_accessed
        for field in fields_accessed:
            layout = get_layout_tuple(field)
            permuted_shape = list(field.shape[i] for i in layout)
            self.set_variable(field.name, str(field.dtype), tuple(permuted_shape))

        # Scalars may be safely ignored
        # for param in ast.get_parameters():
        #     if not param.is_field_parameter:
        #         # self.set_variable(param.symbol.name, str(param.symbol.dtype), None)
        #         self.sources[param.symbol.name] = [None]

        # data type
        self.datatype = list(self.variables.values())[0][0]

        # flops
        operation_count = count_operations_in_ast(inner_loop)
        self._flops = {
            '+': operation_count['adds'],
            '*': operation_count['muls'],
            '/': operation_count['divs'],
        }
        for k in [k for k, v in self._flops.items() if v == 0]:
            del self._flops[k]
        self.check()

        if debug_print:
            from pprint import pprint
            print("-----------------------------  Loop Stack --------------------------")
            pprint(self._loop_stack)
            print("-----------------------------  Sources -----------------------------")
            pprint(self.sources)
            print("-----------------------------  Destinations ------------------------")
            pprint(self.destinations)
            print("-----------------------------  FLOPS -------------------------------")
            pprint(self._flops)

    def as_code(self, type_='iaca', openmp=False, as_filename=False):
        """
        Generate and return compilable source code.

        Args:
            type_: can be iaca or likwid.
            openmp: if true, openmp code will be generated
            as_filename:
        """
        code = generate_benchmark(self.kernel_ast, likwid=type_ == 'likwid', openmp=openmp)
        if as_filename:
            fp, already_available = self._get_intermediate_file('kernel_{}.c'.format(type_),
                                                                machine_and_compiler_dependent=False)
            if not already_available:
                fp.write(code)
            return fp.name
        else:
            return code


class KerncraftParameters(DotDict):
    def __init__(self, **kwargs):
        super(KerncraftParameters, self).__init__(**kwargs)
        self['asm_block'] = 'auto'
        self['asm_increment'] = 0
        self['cores'] = 1
        self['cache_predictor'] = 'SIM'
        self['verbose'] = 0
        self['pointer_increment'] = 'auto'
        self['iterations'] = 10
        self['unit'] = 'cy/CL'
        self['ignore_warnings'] = True


# ------------------------------------------- Helper functions ---------------------------------------------------------


def search_resolved_field_accesses_in_ast(ast):
    def visit(node, reads, writes):
        if not isinstance(node, SympyAssignment):
            for a in node.args:
                visit(a, reads, writes)
            return

        for expr, accesses in [(node.lhs, writes), (node.rhs, reads)]:
            accesses.update(expr.atoms(ResolvedFieldAccess))

    read_accesses = set()
    write_accesses = set()
    visit(ast, read_accesses, write_accesses)
    return read_accesses, write_accesses
