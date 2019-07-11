import os
import subprocess

from jinja2 import Template

from pystencils.astnodes import PragmaBlock
from pystencils.backends.cbackend import generate_c, get_headers
from pystencils.cpu.cpujit import get_compiler_config, run_compile_step
from pystencils.data_types import get_base_type
from pystencils.include import get_pystencils_include_path
from pystencils.sympyextensions import prod

benchmark_template = Template("""
#include "kerncraft.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

{{ includes }}

{%- if likwid %}
#include <likwid.h>
{%- endif %}

#define RESTRICT __restrict__
#define FUNC_PREFIX
void dummy(void *);
void timing(double* wcTime, double* cpuTime);
extern int var_false;


{{kernel_code}}


int main(int argc, char **argv)
{
  {%- if likwid %}
  likwid_markerInit();
  {%- endif %}

  {%- for field_name, dataType, size in fields %}

  // Initialization {{field_name}}
  double * {{field_name}} = (double *) aligned_malloc(sizeof({{dataType}}) * {{size}}, 64);
  for (unsigned long long i = 0; i < {{size}}; ++i)
    {{field_name}}[i] = 0.23;

  if(var_false)
    dummy({{field_name}});

  {%- endfor %}



  {%- for constantName, dataType in constants %}

  // Constant {{constantName}}
  {{dataType}} {{constantName}};
  {{constantName}} = 0.23;
  if(var_false)
      dummy(& {{constantName}});

  {%- endfor %}

  {%- if likwid and openmp %}
  #pragma omp parallel
  {
  likwid_markerRegisterRegion("loop");
  #pragma omp barrier
  {%- elif likwid %}
  likwid_markerRegisterRegion("loop");
  {%- endif %}

  for(int warmup = 1; warmup >= 0; --warmup) {
    int repeat = 2;
    if(warmup == 0) {
      repeat = atoi(argv[1]);
      {%- if likwid %}
      likwid_markerStartRegion("loop");
      {%- endif %}
    }
    
    {%- if timing %}
    double wcStartTime, cpuStartTime, wcEndTime, cpuEndTime;
    timing(&wcStartTime, &cpuStartTime);
    {%- endif %}
    
    for (; repeat > 0; --repeat)
    {
      {{kernelName}}({{call_argument_list}});

      // Dummy calls
      {%- for field_name, dataType, size in fields %}
      if(var_false) dummy((void*){{field_name}});
      {%- endfor %}
      {%- for constantName, dataType in constants %}
      if(var_false) dummy((void*)&{{constantName}});
      {%- endfor %}
    }
    {%- if timing %}
    timing(&wcEndTime, &cpuEndTime);
    if( warmup == 0)
        printf("%e\\n", (wcEndTime - wcStartTime) / atoi(argv[1]) );
    {%- endif %}

  }

  {%- if likwid %}
  likwid_markerStopRegion("loop");
  {%- if openmp %}
  }
  {%- endif %}
  {%- endif %}

  {%- if likwid %}
  likwid_markerClose();
  {%- endif %}
}
""")


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
            fields.append((p.field_name, dtype, prod(field.shape)))
            call_parameters.append(p.field_name)

    header_list = get_headers(ast)
    includes = "\n".join(["#include %s" % (include_file,) for include_file in header_list])

    # Strip "#pragma omp parallel" from within kernel, because main function takes care of that
    # when likwid and openmp are enabled
    if likwid and openmp:
        if len(ast.body.args) > 0 and isinstance(ast.body.args[0], PragmaBlock):
            ast.body.args[0].pragma_line = ''

    args = {
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
    return benchmark_template.render(**args)


def run_c_benchmark(ast, inner_iterations, outer_iterations=3):
    """Runs the given kernel with outer loop in C

    Args:
        ast:
        inner_iterations: timings are recorded around this many iterations
        outer_iterations: number of timings recorded

    Returns:
        list of times per iterations for each outer iteration
    """
    import kerncraft

    benchmark_code = generate_benchmark(ast, timing=True)
    with open('bench.c', 'w') as f:
        f.write(benchmark_code)

    kerncraft_path = os.path.dirname(kerncraft.__file__)

    extra_flags = ['-I' + get_pystencils_include_path(),
                   '-I' + os.path.join(kerncraft_path, 'headers')]

    compiler_config = get_compiler_config()
    compile_cmd = [compiler_config['command']] + compiler_config['flags'].split()
    compile_cmd += [*extra_flags,
                    os.path.join(kerncraft_path, 'headers', 'timing.c'),
                    os.path.join(kerncraft_path, 'headers', 'dummy.c'),
                    'bench.c',
                    '-o', 'bench',
                    ]
    run_compile_step(compile_cmd)

    results = []
    for _ in range(outer_iterations):
        benchmark_time = float(subprocess.check_output(['./bench', str(inner_iterations)]))
        results.append(benchmark_time)
    return results
