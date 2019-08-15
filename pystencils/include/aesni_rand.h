#if !defined(__AES__) || !defined(__SSE2__)
#error AES-NI and SSE2 need to be enabled
#endif

#include <emmintrin.h> // SSE2
#include <wmmintrin.h> // AES
#ifdef __AVX512VL__
#include <immintrin.h> // AVX*
#endif
#include <cstdint>

#define QUALIFIERS inline
#define TWOPOW53_INV_DOUBLE (1.1102230246251565e-16)
#define TWOPOW32_INV_FLOAT (2.3283064e-10f)

typedef std::uint32_t uint32;
typedef std::uint64_t uint64;

QUALIFIERS __m128i aesni1xm128i(const __m128i & in, const __m128i & k) {
    __m128i x = _mm_xor_si128(k, in);
    x = _mm_aesenc_si128(x, k);     // 1
    x = _mm_aesenc_si128(x, k);     // 2
    x = _mm_aesenc_si128(x, k);     // 3
    x = _mm_aesenc_si128(x, k);     // 4
    x = _mm_aesenc_si128(x, k);     // 5
    x = _mm_aesenc_si128(x, k);     // 6
    x = _mm_aesenc_si128(x, k);     // 7
    x = _mm_aesenc_si128(x, k);     // 8
    x = _mm_aesenc_si128(x, k);     // 9
    x = _mm_aesenclast_si128(x, k); // 10
    return x;
}

QUALIFIERS __m128 _my_cvtepu32_ps(const __m128i v)
{
#ifdef __AVX512VL__
    return _mm_cvtepu32_ps(v);
#else
    __m128i v2 = _mm_srli_epi32(v, 1);
    __m128i v1 = _mm_and_si128(v, _mm_set1_epi32(1));
    __m128 v2f = _mm_cvtepi32_ps(v2);
    __m128 v1f = _mm_cvtepi32_ps(v1);
    return _mm_add_ps(_mm_add_ps(v2f, v2f), v1f);
#endif
}

QUALIFIERS __m128d _my_cvtepu64_pd(const __m128i x)
{
#ifdef __AVX512VL__
    return _mm_cvtepu64_pd(x);
#else
    uint64 r[2];
    _mm_storeu_si128((__m128i*)r, x);
    return _mm_set_pd((double)r[1], (double)r[0]);
#endif
}


QUALIFIERS void aesni_double2(uint32 ctr0, uint32 ctr1, uint32 ctr2, uint32 ctr3,
                              uint32 key0, uint32 key1, uint32 key2, uint32 key3,
                              double & rnd1, double & rnd2)
{
    // pack input and call AES
    __m128i c128 = _mm_set_epi32(ctr3, ctr2, ctr1, ctr0);
    __m128i k128 = _mm_set_epi32(key3, key2, key1, key0);
    c128 = aesni1xm128i(c128, k128);

    // convert 32 to 64 bit and put 0th and 2nd element into x, 1st and 3rd element into y
    __m128i x = _mm_and_si128(c128, _mm_set_epi32(0, 0xffffffff, 0, 0xffffffff));
    __m128i y = _mm_and_si128(c128, _mm_set_epi32(0xffffffff, 0, 0xffffffff, 0));
    y = _mm_srli_si128(y, 4);

    // calculate z = x ^ y << (53 - 32))
    __m128i z = _mm_sll_epi64(y, _mm_set_epi64x(53 - 32, 53 - 32));
    z = _mm_xor_si128(x, z);

    // convert uint64 to double
    __m128d rs = _my_cvtepu64_pd(z);
    // calculate rs * TWOPOW53_INV_DOUBLE + (TWOPOW53_INV_DOUBLE/2.0)
    rs = _mm_mul_pd(rs, _mm_set_pd1(TWOPOW53_INV_DOUBLE));
    rs = _mm_add_pd(rs, _mm_set_pd1(TWOPOW53_INV_DOUBLE/2.0));

    // store result
    double rr[2];
    _mm_storeu_pd(rr, rs);
    rnd1 = rr[0];
    rnd2 = rr[1];
}


QUALIFIERS void aesni_float4(uint32 ctr0, uint32 ctr1, uint32 ctr2, uint32 ctr3,
                             uint32 key0, uint32 key1, uint32 key2, uint32 key3,
                             float & rnd1, float & rnd2, float & rnd3, float & rnd4)
{
    // pack input and call AES
    __m128i c128 = _mm_set_epi32(ctr3, ctr2, ctr1, ctr0);
    __m128i k128 = _mm_set_epi32(key3, key2, key1, key0);
    c128 = aesni1xm128i(c128, k128);

    // convert uint32 to float
    __m128 rs = _my_cvtepu32_ps(c128);
    // calculate rs * TWOPOW32_INV_FLOAT + (TWOPOW32_INV_FLOAT/2.0f)
    rs = _mm_mul_ps(rs, _mm_set_ps1(TWOPOW32_INV_FLOAT));
    rs = _mm_add_ps(rs, _mm_set_ps1(TWOPOW32_INV_FLOAT/2.0f));

    // store result
    float r[4];
    _mm_storeu_ps(r, rs);
    rnd1 = r[0];
    rnd2 = r[1];
    rnd3 = r[2];
    rnd4 = r[3];
}