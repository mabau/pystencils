#include <arm_neon.h>

inline float32x4_t makeVec_f32(float a, float b, float c, float d)
{
    alignas(16) float data[4] = {a, b, c, d};
    return vld1q_f32(data);
}

inline float64x2_t makeVec_f64(double a, double b)
{
    alignas(16) double data[2] = {a, b};
    return vld1q_f64(data);
}

inline int32x4_t makeVec_s32(int a, int b, int c, int d)
{
    alignas(16) int data[4] = {a, b, c, d};
    return vld1q_s32(data);
}
