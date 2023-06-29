#pragma once

#define POS_INFINITY __int_as_float(0x7f800000)
#define INFINITY POS_INFINITY
#define NEG_INFINITY __int_as_float(0xff800000)

#ifdef __HIPCC_RTC__
typedef __hip_uint8_t uint8_t;
typedef __hip_int8_t int8_t;
typedef __hip_uint16_t uint16_t;
typedef __hip_int16_t int16_t;
#endif
