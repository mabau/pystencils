#if !defined(__AES__) || !defined(__SSE2__)
#error AES-NI and SSE2 need to be enabled
#endif

#include <x86intrin.h>
#include <cstdint>

#define QUALIFIERS inline
#define TWOPOW53_INV_DOUBLE (1.1102230246251565e-16)
#define TWOPOW32_INV_FLOAT (2.3283064e-10f)

typedef std::uint32_t uint32;
typedef std::uint64_t uint64;

QUALIFIERS __m128i aesni1xm128i(const __m128i & in, const __m128i & k) {
    __m128i x = _mm_xor_si128(k, in);
    x = _mm_aesenc_si128(x, k);
    x = _mm_aesenc_si128(x, k);
    x = _mm_aesenc_si128(x, k);
    x = _mm_aesenc_si128(x, k);
    x = _mm_aesenc_si128(x, k);
    x = _mm_aesenc_si128(x, k);
    x = _mm_aesenc_si128(x, k);
    x = _mm_aesenc_si128(x, k);
    x = _mm_aesenc_si128(x, k);
    x = _mm_aesenclast_si128(x, k);
    return x;
}

QUALIFIERS double _uniform_double_hq(uint32 x, uint32 y)
{
    uint64 z = (uint64)x ^ ((uint64)y << (53 - 32));
    return z * TWOPOW53_INV_DOUBLE + (TWOPOW53_INV_DOUBLE/2.0);
}


QUALIFIERS void aesni_double2(uint32 ctr0, uint32 ctr1, uint32 ctr2, uint32 ctr3,
                              uint32 key0, uint32 key1, uint32 key2, uint32 key3,
                              double & rnd1, double & rnd2)
{
    __m128i c128 = _mm_set_epi32(ctr3, ctr2, ctr1, ctr0);
    __m128i k128 = _mm_set_epi32(key3, key2, key1, key0);
    c128 = aesni1xm128i(c128, k128);
    uint32 r[4];
    _mm_storeu_si128((__m128i*)&r[0], c128);

    rnd1 = _uniform_double_hq(r[0], r[1]);
    rnd2 = _uniform_double_hq(r[2], r[3]);
}


QUALIFIERS void aesni_float4(uint32 ctr0, uint32 ctr1, uint32 ctr2, uint32 ctr3,
                             uint32 key0, uint32 key1, uint32 key2, uint32 key3,
                             float & rnd1, float & rnd2, float & rnd3, float & rnd4)
{
    __m128i c128 = _mm_set_epi32(ctr3, ctr2, ctr1, ctr0);
    __m128i k128 = _mm_set_epi32(key3, key2, key1, key0);
    c128 = aesni1xm128i(c128, k128);
    uint32 r[4];
    _mm_storeu_si128((__m128i*)&r[0], c128);

    rnd1 = r[0] * TWOPOW32_INV_FLOAT + (TWOPOW32_INV_FLOAT/2.0f);
    rnd2 = r[1] * TWOPOW32_INV_FLOAT + (TWOPOW32_INV_FLOAT/2.0f);
    rnd3 = r[2] * TWOPOW32_INV_FLOAT + (TWOPOW32_INV_FLOAT/2.0f);
    rnd4 = r[3] * TWOPOW32_INV_FLOAT + (TWOPOW32_INV_FLOAT/2.0f);
}