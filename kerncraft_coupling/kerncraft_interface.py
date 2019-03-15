from tempfile import TemporaryDirectory

import sympy as sp
import os
from collections import defaultdict
import subprocess
import kerncraft
import kerncraft.kernel
from typing import Optional
from kerncraft.iaca import iaca_analyse_instrumented_binary, iaca_instrumentation
from kerncraft.machinemodel import MachineModel

from pystencils.kerncraft_coupling.generate_benchmark import generate_benchmark
from pystencils.astnodes import LoopOverCoordinate, SympyAssignment, ResolvedFieldAccess, KernelFunction
from pystencils.field import get_layout_from_strides
from pystencils.sympyextensions import count_operations_in_ast
from pystencils.utils import DotDict


class PyStencilsKerncraftKernel(kerncraft.kernel.Kernel):
    """
    Implementation of kerncraft's kernel interface for pystencils CPU kernels.
    Analyses a list of equations assuming they will be executed on a CPU
    """
    LIKWID_BASE = '/usr/local/likwid'

    def __init__(self, ast: KernelFunction, machine: Optional[MachineModel] = None, assumed_layout='SoA'):
        """Create a kerncraft kernel using a pystencils AST

        Args:
            ast: pystencils ast
            machine: kerncraft machine model - specify this if kernel needs to be compiled
            assumed_layout: either 'SoA' or 'AoS' - if fields have symbolic sizes the layout of the index coordinates is not
                    known. In this case either a structures of array (SoA) or array of structures (AoS) layout
                    is assumed
        """
        super(PyStencilsKerncraftKernel, self).__init__(machine)

        self.ast = ast
        self.temporary_dir = TemporaryDirectory()

        # Loops
        inner_loops = [l for l in ast.atoms(LoopOverCoordinate) if l.is_innermost_loop]
        if len(inner_loops) == 0:
            raise ValueError("No loop found in pystencils AST")
        elif len(inner_loops) > 1:
            raise ValueError("pystencils AST contains multiple inner loops - only one can be analyzed")
        else:
            inner_loop = inner_loops[0]

        self._loop_stack = []
        cur_node = inner_loop
        while cur_node is not None:
            if isinstance(cur_node, LoopOverCoordinate):
                loop_counter_sym = cur_node.loop_counter_symbol
                loop_info = (loop_counter_sym.name, cur_node.start, cur_node.stop, cur_node.step)
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

        for param in ast.get_parameters():
            if not param.is_field_parameter:
                self.set_variable(param.symbol.name, str(param.symbol.dtype), None)
                self.sources[param.symbol.name] = [None]

        # data type
        self.datatype = list(self.variables.values())[0][0]

        # flops
        # FIXME operation_count
        operation_count = count_operations_in_ast(inner_loop)
        self._flops = {
            '+': operation_count['adds'],
            '*': operation_count['muls'],
            '/': operation_count['divs'],
        }
        for k in [k for k, v in self._flops.items() if v == 0]:
            del self._flops[k]
        self.check()

    def iaca_analysis(self, micro_architecture, asm_block='auto',
                      pointer_increment='auto_with_manual_fallback', verbose=False):
        compiler, compiler_args = self._machine.get_compiler()
        if '-std=c99' not in compiler_args:
            compiler_args += ['-std=c99']
        header_path = kerncraft.get_header_path()

        compiler_cmd = [compiler] + compiler_args + ['-I' + header_path]

        src_file = os.path.join(self.temporary_dir.name, "source.c")
        asm_file = os.path.join(self.temporary_dir.name, "source.s")
        iaca_asm_file = os.path.join(self.temporary_dir.name, "source.iaca.s")
        dummy_src_file = os.path.join(header_path, "dummy.c")
        dummy_asm_file = os.path.join(self.temporary_dir.name, "dummy.s")
        binary_file = os.path.join(self.temporary_dir.name, "binary")

        # write source code to file
        with open(src_file, 'w') as f:
            f.write(generate_benchmark(self.ast, likwid=False))

        # compile to asm files
        subprocess.check_output(compiler_cmd + [src_file, '-S', '-o', asm_file])
        subprocess.check_output(compiler_cmd + [dummy_src_file, '-S', '-o', dummy_asm_file])

        with open(asm_file) as read, open(iaca_asm_file, 'w') as write:
            instrumented_asm_block = iaca_instrumentation(read, write)

        # assemble asm files to executable
        subprocess.check_output(compiler_cmd + [iaca_asm_file, dummy_asm_file, '-o', binary_file])

        result = iaca_analyse_instrumented_binary(binary_file, micro_architecture)
    
        return result, instrumented_asm_block

    def build(self, lflags=None, verbose=False, openmp=False):
        # TODO do we use openmp or not???
        compiler, compiler_args = self._machine.get_compiler()
        if '-std=c99' not in compiler_args:
            compiler_args.append('-std=c99')
        header_path = kerncraft.get_header_path()

        cmd = [compiler] + compiler_args + [
            '-I' + os.path.join(self.LIKWID_BASE, 'include'),
            '-L' + os.path.join(self.LIKWID_BASE, 'lib'),
            '-I' + header_path,
            '-Wl,-rpath=' + os.path.join(self.LIKWID_BASE, 'lib'),
        ]

        dummy_src_file = os.path.join(header_path, 'dummy.c')
        src_file = os.path.join(self.temporary_dir.name, "source_likwid.c")
        bin_file = os.path.join(self.temporary_dir.name, "benchmark")

        with open(src_file, 'w') as f:
            f.write(generate_benchmark(self.ast, likwid=True))

        subprocess.check_output(cmd + [src_file, dummy_src_file, '-pthread', '-llikwid', '-o', bin_file])
        return bin_file


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
