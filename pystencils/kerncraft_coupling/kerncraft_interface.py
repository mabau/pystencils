import warnings
import fcntl
from collections import defaultdict
from tempfile import TemporaryDirectory
import textwrap
import itertools
import string

from jinja2 import Environment, PackageLoader, StrictUndefined, Template
import sympy as sp
from kerncraft.kerncraft import KernelCode
from kerncraft.kernel import symbol_pos_int
from kerncraft.machinemodel import MachineModel

from pystencils.astnodes import \
    KernelFunction, LoopOverCoordinate, ResolvedFieldAccess, SympyAssignment
from pystencils.backends.cbackend import generate_c, get_headers
from pystencils.field import get_layout_from_strides
from pystencils.sympyextensions import count_operations_in_ast
from pystencils.transformations import filtered_tree_iteration
from pystencils.utils import DotDict
from pystencils.cpu.kernelcreation import add_openmp
from pystencils.data_types import get_base_type
from pystencils.sympyextensions import prod


class PyStencilsKerncraftKernel(KernelCode):
    """
    Implementation of kerncraft's kernel interface for pystencils CPU kernels.
    Analyses a list of equations assuming they will be executed on a CPU
    """
    LIKWID_BASE = '/usr/local/likwid'

    def __init__(self, ast: KernelFunction, machine: MachineModel,
                 assumed_layout='SoA', debug_print=False, filename=None):
        """Create a kerncraft kernel using a pystencils AST

        Args:
            ast: pystencils ast
            machine: kerncraft machine model - specify this if kernel needs to be compiled
            assumed_layout: either 'SoA' or 'AoS' - if fields have symbolic sizes the layout of the index
                    coordinates is not known. In this case either a structures of array (SoA) or
                    array of structures (AoS) layout is assumed
            debug_print: print debug information
            filename: used for caching
        """
        super(KernelCode, self).__init__(machine=machine)

        # Initialize state
        self.asm_block = None
        self._filename = filename
        self._keep_intermediates = False

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

        def get_layout_tuple(f):
            if f.has_fixed_shape:
                return get_layout_from_strides(f.strides)
            else:
                layout_list = list(f.layout)
                for _ in range(f.index_dimensions):
                    layout_list.insert(0 if assumed_layout == 'SoA' else -1, max(layout_list) + 1)
                return layout_list

        # Variables (arrays) and Constants (scalar sizes)
        const_names_iter = itertools.product(string.ascii_uppercase, repeat=1)
        constants_reversed = {}
        fields_accessed = self.kernel_ast.fields_accessed
        for field in fields_accessed:
            layout = get_layout_tuple(field)
            permuted_shape = list(field.shape[i] for i in layout)
            # Replace shape dimensions with constant variables (necessary for layer condition
            # analysis)
            for i, d in enumerate(permuted_shape):
                if d not in self.constants.values():
                    const_symbol = symbol_pos_int(''.join(next(const_names_iter)))
                    self.set_constant(const_symbol, d)
                    constants_reversed[d] = const_symbol
                permuted_shape[i] = constants_reversed[d]
            self.set_variable(field.name, (str(field.dtype),), tuple(permuted_shape))

        # Data sources & destinations
        self.sources = defaultdict(list)
        self.destinations = defaultdict(list)

        reads, writes = search_resolved_field_accesses_in_ast(inner_loop)
        for accesses, target_dict in [(reads, self.sources), (writes, self.destinations)]:
            for fa in accesses:
                coord = [symbol_pos_int(LoopOverCoordinate.get_loop_counter_name(i)) + off
                         for i, off in enumerate(fa.offsets)]
                coord += list(fa.idx_coordinate_values)
                layout = get_layout_tuple(fa.field)
                permuted_coord = [sp.sympify(coord[i]) for i in layout]
                target_dict[fa.field.name].append(permuted_coord)

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

    def get_kernel_header(self, name='pystencils_kernel'):
        file_name = "pystencils_kernel.h"
        file_path = self.get_intermediate_location(file_name, machine_and_compiler_dependent=False)
        lock_mode, lock_fp = self.lock_intermediate(file_path)

        if lock_mode == fcntl.LOCK_SH:
            # use cache
            pass
        else:  # lock_mode == fcntl.LOCK_EX:
            function_signature = generate_c(self.kernel_ast, dialect='c', signature_only=True)

            jinja_context = {
                'function_signature': function_signature,
            }

            env = Environment(loader=PackageLoader('pystencils.kerncraft_coupling'), undefined=StrictUndefined)
            file_header = env.get_template('kernel.h').render(**jinja_context)
            with open(file_path, 'w') as f:
                f.write(file_header)

            self.release_exclusive_lock(lock_fp)  # degrade to shared lock
        return file_path, lock_fp

    def get_kernel_code(self, openmp=False, name='pystencils_kernl'):
        """
        Generate and return compilable source code from AST.

        Args:
            openmp: if true, openmp code will be generated
            name: kernel name
        """
        filename = 'pystencils_kernl'
        if openmp:
            filename += '-omp'
        filename += '.c'
        file_path = self.get_intermediate_location(filename, machine_and_compiler_dependent=False)
        lock_mode, lock_fp = self.lock_intermediate(file_path)

        if lock_mode == fcntl.LOCK_SH:
            # use cache
            with open(file_path) as f:
                code = f.read()
        else:  # lock_mode == fcntl.LOCK_EX:
            header_list = get_headers(self.kernel_ast)
            includes = "\n".join(["#include %s" % (include_file,) for include_file in header_list])

            if openmp:
                add_openmp(self.kernel_ast)

            kernel_code = generate_c(self.kernel_ast, dialect='c')

            jinja_context = {
                'includes': includes,
                'kernel_code': kernel_code,
            }

            env = Environment(loader=PackageLoader('pystencils.kerncraft_coupling'), undefined=StrictUndefined)
            code = env.get_template('kernel.c').render(**jinja_context)
            with open(file_path, 'w') as f:
                f.write(code)

            self.release_exclusive_lock(lock_fp)  # degrade to shared lock
        return file_path, lock_fp

    CODE_TEMPLATE = Template(textwrap.dedent("""
        #include <likwid.h>
        #include <stdlib.h>
        #include <stdint.h>
        #include <stdbool.h>
        #include <math.h>
        #include "kerncraft.h"
        #include "kernel.h"

        #define RESTRICT __restrict__
        #define FUNC_PREFIX
        void dummy(void *);
        extern int var_false;

        int main(int argc, char **argv) {
          {%- for constantName, dataType in constants %}
          // Constant {{constantName}}
          {{dataType}} {{constantName}};
          {{constantName}} = 0.23;
          {%- endfor %}

          // Declaring arrays
          {%- for field_name, dataType, size in fields %}

          // Initialization {{field_name}}
          double * {{field_name}} = (double *) aligned_malloc(sizeof({{dataType}}) * {{size}}, 64);
          // TODO initialize in parallel context in same order as they are touched
          for (unsigned long long i = 0; i < {{size}}; ++i)
            {{field_name}}[i] = 0.23;
          {%- endfor %}

          likwid_markerInit();
          #pragma omp parallel
          {
            likwid_markerRegisterRegion("loop");
            #pragma omp barrier

            // Initializing arrays in same order as touched in kernel loop nest
            //INIT_ARRAYS;

            // Dummy call
            {%- for field_name, dataType, size in fields %}
            if(var_false) dummy({{field_name}});
            {%- endfor %}
            {%- for constantName, dataType in constants %}
            if(var_false) dummy(&{{constantName}});
            {%- endfor %}

            for(int warmup = 1; warmup >= 0; --warmup) {
              int repeat = 2;
              if(warmup == 0) {
                repeat = atoi(argv[1]);
                likwid_markerStartRegion("loop");
              }

              for(; repeat > 0; --repeat) {
                {{kernelName}}({{call_argument_list}});

                {%- for field_name, dataType, size in fields %}
                if(var_false) dummy({{field_name}});
                {%- endfor %}
                {%- for constantName, dataType in constants %}
                if(var_false) dummy(&{{constantName}});
                {%- endfor %}
              }

            }
            likwid_markerStopRegion("loop");
          }
          likwid_markerClose();
          return 0;
        }
        """))

    def get_main_code(self, kernel_function_name='kernel'):
        """
        Generate and return compilable source code from AST.

        :return: tuple of filename and shared lock file pointer
        """
        # TODO produce nicer code, including help text and other "comfort features".
        assert self.kernel_ast is not None, "AST does not exist, this could be due to running " \
                                            "based on a kernel description rather than code."

        file_path = self.get_intermediate_location('main.c', machine_and_compiler_dependent=False)
        lock_mode, lock_fp = self.lock_intermediate(file_path)

        if lock_mode == fcntl.LOCK_SH:
            # use cache
            with open(file_path) as f:
                code = f.read()
        else:  # lock_mode == fcntl.LOCK_EX
            # needs update
            accessed_fields = {f.name: f for f in self.kernel_ast.fields_accessed}
            constants = []
            fields = []
            call_parameters = []
            for p in self.kernel_ast.get_parameters():
                if not p.is_field_parameter:
                    constants.append((p.symbol.name, str(p.symbol.dtype)))
                    call_parameters.append(p.symbol.name)
                else:
                    assert p.is_field_pointer, "Benchmark implemented only for kernels with fixed loop size"
                    field = accessed_fields[p.field_name]
                    dtype = str(get_base_type(p.symbol.dtype))
                    fields.append((p.field_name, dtype, prod(field.shape)))
                    call_parameters.append(p.field_name)

            header_list = get_headers(self.kernel_ast)
            includes = "\n".join(["#include %s" % (include_file,) for include_file in header_list])

            # Generate code
            code = self.CODE_TEMPLATE.render(
                kernelName=self.kernel_ast.function_name,
                fields=fields,
                constants=constants,
                call_agument_list=','.join(call_parameters),
                includes=includes)

            # Store to file
            with open(file_path, 'w') as f:
                f.write(code)
            self.release_exclusive_lock(lock_fp)  # degrade to shared lock

        return file_path, lock_fp


class KerncraftParameters(DotDict):
    def __init__(self, **kwargs):
        super(KerncraftParameters, self).__init__()
        self['asm_block'] = 'auto'
        self['asm_increment'] = 0
        self['cores'] = 1
        self['cache_predictor'] = 'SIM'
        self['verbose'] = 0
        self['pointer_increment'] = 'auto'
        self['iterations'] = 10
        self['unit'] = 'cy/CL'
        self['ignore_warnings'] = True
        self['incore_model'] = 'OSACA'
        self.update(**kwargs)


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
