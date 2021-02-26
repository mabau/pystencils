import subprocess
import warnings
import tempfile
from pathlib import Path

from jinja2 import Environment, PackageLoader, StrictUndefined

from pystencils.astnodes import PragmaBlock
from pystencils.backends.cbackend import generate_c, get_headers
from pystencils.cpu.cpujit import get_compiler_config, run_compile_step
from pystencils.data_types import get_base_type
from pystencils.include import get_pystencils_include_path
from pystencils.integer_functions import modulo_ceil
from pystencils.sympyextensions import prod

import numpy as np


def generate_benchmark(ast, likwid=False, openmp=False, timing=False):
    """Return C code of a benchmark program for the given kernel.

    Args:
        ast: the pystencils AST object as returned by create_kernel
        likwid: if True likwid markers are added to the code
        openmp: relevant only if likwid=True, to generated correct likwid initialization code
        timing: add timing output to the code, prints time per iteration to stdout

    Returns:
        C code as string
    """
    accessed_fields = {f.name: f for f in ast.fields_accessed}
    constants = []
    fields = []
    call_parameters = []
    for p in ast.get_parameters():
        if not p.is_field_parameter:
            constants.append((p.symbol.name, str(p.symbol.dtype)))
            call_parameters.append(p.symbol.name)
        else:
            assert p.is_field_pointer, "Benchmark implemented only for kernels with fixed loop size"
            field = accessed_fields[p.field_name]
            dtype = str(get_base_type(p.symbol.dtype))
            np_dtype = get_base_type(p.symbol.dtype).numpy_dtype
            size_data_type = np_dtype.itemsize

            dim0_size = field.shape[-1]
            dim1_size = np.prod(field.shape[:-1])
            elements = prod(field.shape)

            if ast.instruction_set:
                align = ast.instruction_set['width'] * size_data_type
                padding_elements = modulo_ceil(dim0_size, ast.instruction_set['width']) - dim0_size
                padding_bytes = padding_elements * size_data_type
                ghost_layers = max(max(ast.ghost_layers))

                size = dim1_size * padding_bytes + np.prod(field.shape) * size_data_type

                assert align % np_dtype.itemsize == 0
                offset = ((dim0_size + padding_elements + ghost_layers) % ast.instruction_set['width']) * size_data_type

                fields.append((p.field_name, dtype, elements, size, offset, align))
                call_parameters.append(p.field_name)
            else:
                size = elements * size_data_type
                fields.append((p.field_name, dtype, elements, size, 0, 0))
                call_parameters.append(p.field_name)

    header_list = get_headers(ast)
    includes = "\n".join(["#include %s" % (include_file,) for include_file in header_list])

    # Strip "#pragma omp parallel" from within kernel, because main function takes care of that
    # when likwid and openmp are enabled
    if likwid and openmp:
        if len(ast.body.args) > 0 and isinstance(ast.body.args[0], PragmaBlock):
            ast.body.args[0].pragma_line = ''

    jinja_context = {
        'likwid': likwid,
        'openmp': openmp,
        'kernel_code': generate_c(ast, dialect='c'),
        'kernelName': ast.function_name,
        'fields': fields,
        'constants': constants,
        'call_argument_list': ",".join(call_parameters),
        'includes': includes,
        'timing': timing,
    }

    env = Environment(loader=PackageLoader('pystencils.kerncraft_coupling'), undefined=StrictUndefined)

    return env.get_template('benchmark.c').render(**jinja_context)


def run_c_benchmark(ast, inner_iterations, outer_iterations=3, path=None):
    """Runs the given kernel with outer loop in C

    Args:
        ast: pystencils ast which is used to compile the benchmark file
        inner_iterations: timings are recorded around this many iterations
        outer_iterations: number of timings recorded
        path: path where the benchmark file is stored. If None a tmp folder is created

    Returns:
        list of times per iterations for each outer iteration
    """
    import kerncraft

    benchmark_code = generate_benchmark(ast, timing=True)

    if path is None:
        path = tempfile.mkdtemp()

    if isinstance(path, str):
        path = Path(path)

    with open(path / 'bench.c', 'w') as f:
        f.write(benchmark_code)

    kerncraft_path = Path(kerncraft.__file__).parent

    extra_flags = ['-I' + get_pystencils_include_path(),
                   '-I' + str(kerncraft_path / 'headers')]

    compiler_config = get_compiler_config()
    compile_cmd = [compiler_config['command']] + compiler_config['flags'].split()
    compile_cmd += [*extra_flags,
                    str(kerncraft_path / 'headers' / 'timing.c'),
                    str(kerncraft_path / 'headers' / 'dummy.c'),
                    str(path / 'bench.c'),
                    '-o', str(path / 'bench'),
                    ]
    run_compile_step(compile_cmd)

    time_pre_estimation_per_iteration = float(subprocess.check_output(['./' / path / 'bench', str(10)]))
    benchmark_time_limit = 20
    if benchmark_time_limit / time_pre_estimation_per_iteration < inner_iterations:
        warn = (f"A benchmark run with {inner_iterations} inner_iterations will probably take longer than "
                f"{benchmark_time_limit} seconds for this kernel")
        warnings.warn(warn)

    results = []
    for _ in range(outer_iterations):
        benchmark_time = float(subprocess.check_output(['./' / path / 'bench', str(inner_iterations)]))
        results.append(benchmark_time)
    return results
