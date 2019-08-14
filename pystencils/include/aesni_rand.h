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

QUALIFIERS __m128 _my_cvtepu32_ps(const __m128i v)
{
#ifdef __AVX512F__
    return _mm_cvtepu32_ps(v);
#else
    __m128i v2 = _mm_srli_epi32(v, 1);
    __m128i v1 = _mm_and_si128(v, _mm_set1_epi32(1));
    __m128 v2f = _mm_cvtepi32_ps(v2);
    __m128 v1f = _mm_cvtepi32_ps(v1);
    return _mm_add_ps(_mm_add_ps(v2f, v2f), v1f);
#endif
}

QUALIFIERS __m128d _my_cvtepu64_pd(const __m128i v)
{
#ifdef __AVX512F__
    return _mm_cvtepu64_pd(v);
#else
    #warning need to implement _my_cvtepu64_pd
    return (__m128d) v;
#endif
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
    __m128i x = _mm_set_epi64x((uint64) r[2], (uint64) r[0]);
    __m128i y = _mm_set_epi64x((uint64) r[3], (uint64) r[1]);

    __m128i cnt = _mm_set_epi64x(53 - 32, 53 - 32);
    y = _mm_sll_epi64(y, cnt);
    __m128i z = _mm_xor_si128(x, y);

    __m128d rs = _my_cvtepu64_pd(z);
    const __m128d tp53 = _mm_set_pd1(TWOPOW53_INV_DOUBLE);
    const __m128d tp54 = _mm_set_pd1(TWOPOW53_INV_DOUBLE/2.0);
    rs = _mm_mul_pd(rs, tp53);
    rs = _mm_add_pd(rs, tp54);

    double rr[2];
    _mm_storeu_pd(rr, rs);
    rnd1 = rr[0];
    rnd2 = rr[1];
}


QUALIFIERS void aesni_float4(uint32 ctr0, uint32 ctr1, uint32 ctr2, uint32 ctr3,
                             uint32 key0, uint32 key1, uint32 key2, uint32 key3,
                             float & rnd1, float & rnd2, float & rnd3, float & rnd4)
{
    __m128i c128 = _mm_set_epi32(ctr3, ctr2, ctr1, ctr0);
    __m128i k128 = _mm_set_epi32(key3, key2, key1, key0);
    c128 = aesni1xm128i(c128, k128);

    __m128 rs = _my_cvtepu32_ps(c128);
    const __m128 tp32 = _mm_set_ps1(TWOPOW32_INV_FLOAT);
    const __m128 tp33 = _mm_set_ps1(TWOPOW32_INV_FLOAT/2.0f);
    rs = _mm_mul_ps(rs, tp32);
    rs = _mm_add_ps(rs, tp33);

    float r[4];
    _mm_storeu_ps(r, rs);
    rnd1 = r[0];
    rnd2 = r[1];
    rnd3 = r[2];
    rnd4 = r[3];
}