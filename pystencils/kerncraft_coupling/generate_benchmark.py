from jinja2 import Template
from pystencils.backends.cbackend import generate_c, get_headers
from pystencils.sympyextensions import prod
from pystencils.data_types import get_base_type
from pystencils.astnodes import PragmaBlock

benchmark_template = Template("""
#include "kerncraft.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
{{ includes }}

{%- if likwid %}
#include <likwid.h>
{%- endif %}

#define RESTRICT __restrict__
#define FUNC_PREFIX
void dummy(double *);
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

    for (; repeat > 0; --repeat)
    {
      {{kernelName}}({{call_argument_list}});

      // Dummy calls
      {%- for field_name, dataType, size in fields %}
      if(var_false) dummy({{field_name}});
      {%- endfor %}
      {%- for constantName, dataType in constants %}
      if(var_false) dummy(&{{constantName}});
      {%- endfor %}
    }
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


def generate_benchmark(ast, likwid=False, openmp=False):
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
    }
    return benchmark_template.render(**args)
