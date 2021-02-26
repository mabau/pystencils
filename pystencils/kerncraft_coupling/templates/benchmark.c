#include "kerncraft.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

{{ includes }}

{%- if likwid %}
#include <likwid.h>
{%- endif %}

#define RESTRICT __restrict__
#define FUNC_PREFIX
void dummy(void *);
void timing(double* wcTime, double* cpuTime);
extern int var_false;

/* see waLBerla src/field/allocation/AlignedMalloc */
void *aligned_malloc_with_offset( uint64_t size, uint64_t alignment, uint64_t offset )
{
    // With 0 alignment this function makes no sense
    // use normal malloc instead
    assert( alignment > 0 );
    // Tests if alignment is power of two (assuming alignment>0)
    assert( !(alignment & (alignment - 1)) );
    assert( offset < alignment );

    void *pa;  // pointer to allocated memory
    void *ptr; // pointer to usable aligned memory

    pa=std::malloc( (size+2*alignment-1 )+sizeof(void *));
    if(!pa)
        return nullptr;

    // Find next aligned position, starting at pa+sizeof(void*)-1
    ptr=(void*)( ((size_t)pa+sizeof(void *)+alignment-1) & ~(alignment-1));
    ptr=(void*) ( (char*)(ptr) + alignment - offset);

    // Store pointer to real allocated chunk just before usable chunk
    *((void **)ptr-1)=pa;

    assert( ((size_t)ptr+offset) % alignment == 0 );

    return ptr;
}

void aligned_free( void *ptr )
{
    // assume that pointer to real allocated chunk is stored just before
    // chunk that was given to user
    if(ptr)
        std::free(*((void **)ptr-1));
}


{{kernel_code}}


int main(int argc, char **argv)
{
  {%- if likwid %}
  likwid_markerInit();
  {%- endif %}

  {%- for field_name, dataType, elements, size, offset, alignment in fields %}
  // Initialization {{field_name}}
  {%- if alignment > 0 %}
  {{dataType}} * {{field_name}} = ({{dataType}} *) aligned_malloc_with_offset({{size}}, {{alignment}}, {{offset}});
  {%- else %}
  {{dataType}} * {{field_name}} = new {{dataType}}[{{elements}}];
  {%- endif %}
  for (unsigned long long i = 0; i < {{elements}}; ++i)
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
      {%- for field_name, dataType, elements, size, offset, alignment in fields %}
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

  {%- for field_name, dataType, elements, size, offset, alignment in fields %}
  {%- if alignment > 0 %}
  aligned_free({{field_name}});
  {%- else %}
  delete[] {{field_name}};
  {%- endif %}

  {%- endfor %}
}
