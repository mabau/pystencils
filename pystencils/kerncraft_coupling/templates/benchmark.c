
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
        printf("%e\n", (wcEndTime - wcStartTime) / atoi(argv[1]) );
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
