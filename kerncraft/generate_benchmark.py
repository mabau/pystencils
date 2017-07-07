from jinja2 import Template
from pystencils.cpu import generateC
from pystencils.sympyextensions import prod
from pystencils.types import getBaseType

benchmarkTemplate = Template("""
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


{{kernelCode}}


int main(int argc, char **argv)
{
  {%- if likwid %}
  likwid_markerInit();
  likwid_markerThreadInit();
  {%- endif %}

  {%- for fieldName, dataType, size in fields %}
  
  // Initialization {{fieldName}} 
  double * {{fieldName}} = aligned_malloc(sizeof({{dataType}}) * {{size}}, 32);
  for (int i = 0; i < {{size}}; ++i)
    {{fieldName}}[i] = 0.23;
  
  if(var_false)
    dummy({{fieldName}});   
         
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
    kernel({{callArgumentList}});
    
    // Dummy calls   
    {%- for fieldName, dataType, size in fields %}
    if(var_false) dummy({{fieldName}});      
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


def generateBenchmark(ast, likwid=False):
    accessedFields = {f.name: f for f in ast.fieldsAccessed}
    constants = []
    fields = []
    callParameters = []
    for p in ast.parameters:
        if not p.isFieldArgument:
            constants.append((p.name, str(p.dtype)))
            callParameters.append(p.name)
        else:
            assert p.isFieldPtrArgument, "Benchmark implemented only for kernels with fixed loop size"
            field = accessedFields[p.fieldName]
            dtype = str(getBaseType(p.dtype))
            fields.append((p.fieldName, dtype, prod(field.shape)))
            callParameters.append(p.fieldName)

    args = {
        'likwid': likwid,
        'kernelCode': generateC(ast),
        'fields': fields,
        'constants': constants,
        'callArgumentList': ",".join(callParameters),
    }
    return benchmarkTemplate.render(**args)
