from jinja2 import Template
from pystencils.backends.cbackend import generate_c
from pystencils.sympyextensions import prod
from pystencils.data_types import get_base_type

benchmark_template = Template("""
#include "kerncraft.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
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
  likwid_markerThreadInit();
  {%- endif %}

  {%- for field_name, dataType, size in fields %}

  // Initialization {{field_name}}
  double * {{field_name}} = (double *) aligned_malloc(sizeof({{dataType}}) * {{size}}, 32);
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

  int repeat = atoi(argv[1]);
  {%- if likwid %}
  likwid_markerStartRegion("loop");
  {%- endif %}

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

  {%- if likwid %}
  likwid_markerStopRegion("loop");
  {%- endif %}

  {%- if likwid %}
  likwid_markerClose();
  {%- endif %}
}
""")


def generate_benchmark(ast, likwid=False):
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

    args = {
        'likwid': likwid,
        'kernel_code': generate_c(ast),
        'kernelName': ast.function_name,
        'fields': fields,
        'constants': constants,
        'call_argument_list': ",".join(call_parameters),
    }
    return benchmark_template.render(**args)
