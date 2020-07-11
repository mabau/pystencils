
#include "kerncraft.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

{{ includes }}

#define RESTRICT __restrict__
#define FUNC_PREFIX
void dummy(void *);
void timing(double* wcTime, double* cpuTime);
extern int var_false;


{{kernel_code}}