#include <emmintrin.h> // SSE2
#include <wmmintrin.h> // AES
#ifdef __AVX__
#include <immintrin.h> // AVX*
#else
#include <smmintrin.h>  // SSE4
#ifdef __FMA__
#include <immintrin.h> // FMA
#endif
#endif
#include <cstdint>
#include <array>
#include <map>

#define QUALIFIERS inline
#define TWOPOW53_INV_DOUBLE (1.1102230246251565e-16)
#define TWOPOW32_INV_FLOAT (2.3283064e-10f)

#include "myintrin.h"

typedef std::uint32_t uint32;
typedef std::uint64_t uint64;

template <typename T, std::size_t Alignment>
class AlignedAllocator
{
public:
    typedef T value_type;

    template <typename U>
    struct rebind {
        typedef AlignedAllocator<U, Alignment> other;
    };

    T * allocate(const std::size_t n) const {
        if (n == 0) {
            return nullptr;
        }
        void * const p = _mm_malloc(n*sizeof(T), Alignment);
        if (p == nullptr) {
            throw std::bad_alloc();
        }
        return static_cast<T *>(p);
    }

    void deallocate(T * const p, const std::size_t n) const {
        _mm_free(p);
    }
};

template <typename Key, typename T>
using AlignedMap = std::map<Key, T, std::less<Key>, AlignedAllocator<std::pair<const Key, T>, sizeof(Key)>>;

#if defined(__AES__) || defined(_MSC_VER)
QUALIFIERS __m128i aesni_keygen_assist(__m128i temp1, __m128i temp2) {
    __m128i temp3; 
    temp2 = _mm_shuffle_epi32(temp2 ,0xff); 
    temp3 = _mm_slli_si128(temp1, 0x4);
    temp1 = _mm_xor_si128(temp1, temp3);
    temp3 = _mm_slli_si128(temp3, 0x4);
    temp1 = _mm_xor_si128(temp1, temp3);
    temp3 = _mm_slli_si128(temp3, 0x4);
    temp1 = _mm_xor_si128(temp1, temp3);
    temp1 = _mm_xor_si128(temp1, temp2); 
    return temp1; 
}

QUALIFIERS std::array<__m128i,11> aesni_keygen(__m128i k) {
    std::array<__m128i,11> rk;
    __m128i tmp;
    
    rk[0] = k;
    
    tmp = _mm_aeskeygenassist_si128(k, 0x1);
    k = aesni_keygen_assist(k, tmp);
    rk[1] = k;
    
    tmp = _mm_aeskeygenassist_si128(k, 0x2);
    k = aesni_keygen_assist(k, tmp);
    rk[2] = k;
    
    tmp = _mm_aeskeygenassist_si128(k, 0x4);
    k = aesni_keygen_assist(k, tmp);
    rk[3] = k;
    
    tmp = _mm_aeskeygenassist_si128(k, 0x8);
    k = aesni_keygen_assist(k, tmp);
    rk[4] = k;
    
    tmp = _mm_aeskeygenassist_si128(k, 0x10);
    k = aesni_keygen_assist(k, tmp);
    rk[5] = k;
    
    tmp = _mm_aeskeygenassist_si128(k, 0x20);
    k = aesni_keygen_assist(k, tmp);
    rk[6] = k;
    
    tmp = _mm_aeskeygenassist_si128(k, 0x40);
    k = aesni_keygen_assist(k, tmp);
    rk[7] = k;
    
    tmp = _mm_aeskeygenassist_si128(k, 0x80);
    k = aesni_keygen_assist(k, tmp);
    rk[8] = k;
    
    tmp = _mm_aeskeygenassist_si128(k, 0x1b);
    k = aesni_keygen_assist(k, tmp);
    rk[9] = k;
    
    tmp = _mm_aeskeygenassist_si128(k, 0x36);
    k = aesni_keygen_assist(k, tmp);
    rk[10] = k;
    
    return rk;
}

QUALIFIERS const std::array<__m128i,11> & aesni_roundkeys(const __m128i & k128) {
    alignas(16) std::array<uint32,4> a;
    _mm_store_si128((__m128i*) a.data(), k128);
    
    static AlignedMap<std::array<uint32,4>, std::array<__m128i,11>> roundkeys;
    
    if(roundkeys.find(a) == roundkeys.end()) {
        auto rk = aesni_keygen(k128);
        roundkeys[a] = rk;
    }
    return roundkeys[a];
}

QUALIFIERS __m128i aesni1xm128i(const __m128i & in, const __m128i & k0) {
    auto k = aesni_roundkeys(k0);
    __m128i x = _mm_xor_si128(k[0], in);
    x = _mm_aesenc_si128(x, k[1]);
    x = _mm_aesenc_si128(x, k[2]);
    x = _mm_aesenc_si128(x, k[3]);
    x = _mm_aesenc_si128(x, k[4]);
    x = _mm_aesenc_si128(x, k[5]);
    x = _mm_aesenc_si128(x, k[6]);
    x = _mm_aesenc_si128(x, k[7]);
    x = _mm_aesenc_si128(x, k[8]);
    x = _mm_aesenc_si128(x, k[9]);
    x = _mm_aesenclast_si128(x, k[10]);
    return x;
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
    __m128i z = _mm_sll_epi64(y, _mm_set1_epi64x(53 - 32));
    z = _mm_xor_si128(x, z);

    // convert uint64 to double
    __m128d rs = _my_cvtepu64_pd(z);
    // calculate rs * TWOPOW53_INV_DOUBLE + (TWOPOW53_INV_DOUBLE/2.0)
#ifdef __FMA__
    rs = _mm_fmadd_pd(rs, _mm_set1_pd(TWOPOW53_INV_DOUBLE), _mm_set1_pd(TWOPOW53_INV_DOUBLE/2.0));
#else
    rs = _mm_mul_pd(rs, _mm_set1_pd(TWOPOW53_INV_DOUBLE));
    rs = _mm_add_pd(rs, _mm_set1_pd(TWOPOW53_INV_DOUBLE/2.0));
#endif

    // store result
    alignas(16) double rr[2];
    _mm_store_pd(rr, rs);
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
#ifdef __FMA__
    rs = _mm_fmadd_ps(rs, _mm_set1_ps(TWOPOW32_INV_FLOAT), _mm_set1_ps(TWOPOW32_INV_FLOAT/2.0f));
#else
    rs = _mm_mul_ps(rs, _mm_set1_ps(TWOPOW32_INV_FLOAT));
    rs = _mm_add_ps(rs, _mm_set1_ps(TWOPOW32_INV_FLOAT/2.0f));
#endif

    // store result
    alignas(16) float r[4];
    _mm_store_ps(r, rs);
    rnd1 = r[0];
    rnd2 = r[1];
    rnd3 = r[2];
    rnd4 = r[3];
}


template<bool high>
QUALIFIERS __m128d _uniform_double_hq(__m128i x, __m128i y)
{
    // convert 32 to 64 bit
    if (high)
    {
        x = _mm_unpackhi_epi32(x, _mm_setzero_si128());
        y = _mm_unpackhi_epi32(y, _mm_setzero_si128());
    }
    else
    {
        x = _mm_unpacklo_epi32(x, _mm_setzero_si128());
        y = _mm_unpacklo_epi32(y, _mm_setzero_si128());
    }

    // calculate z = x ^ y << (53 - 32))
    __m128i z = _mm_sll_epi64(y, _mm_set1_epi64x(53 - 32));
    z = _mm_xor_si128(x, z);

    // convert uint64 to double
    __m128d rs = _my_cvtepu64_pd(z);
    // calculate rs * TWOPOW53_INV_DOUBLE + (TWOPOW53_INV_DOUBLE/2.0)
#ifdef __FMA__
    rs = _mm_fmadd_pd(rs, _mm_set1_pd(TWOPOW53_INV_DOUBLE), _mm_set1_pd(TWOPOW53_INV_DOUBLE/2.0));
#else
    rs = _mm_mul_pd(rs, _mm_set1_pd(TWOPOW53_INV_DOUBLE));
    rs = _mm_add_pd(rs, _mm_set1_pd(TWOPOW53_INV_DOUBLE/2.0));
#endif

    return rs;
}


QUALIFIERS void aesni_float4(__m128i ctr0, __m128i ctr1, __m128i ctr2, __m128i ctr3,
                              uint32 key0, uint32 key1, uint32 key2, uint32 key3,
                              __m128 & rnd1, __m128 & rnd2, __m128 & rnd3, __m128 & rnd4)
{
    // pack input and call AES
    __m128i k128 = _mm_set_epi32(key3, key2, key1, key0);
    __m128i ctr[4] = {ctr0, ctr1, ctr2, ctr3};
    _MY_TRANSPOSE4_EPI32(ctr[0], ctr[1], ctr[2], ctr[3]);
    for (int i = 0; i < 4; ++i)
    {
        ctr[i] = aesni1xm128i(ctr[i], k128);
    }
    _MY_TRANSPOSE4_EPI32(ctr[0], ctr[1], ctr[2], ctr[3]);

    // convert uint32 to float
    rnd1 = _my_cvtepu32_ps(ctr[0]);
    rnd2 = _my_cvtepu32_ps(ctr[1]);
    rnd3 = _my_cvtepu32_ps(ctr[2]);
    rnd4 = _my_cvtepu32_ps(ctr[3]);
    // calculate rnd * TWOPOW32_INV_FLOAT + (TWOPOW32_INV_FLOAT/2.0f)
#ifdef __FMA__
    rnd1 = _mm_fmadd_ps(rnd1, _mm_set1_ps(TWOPOW32_INV_FLOAT), _mm_set1_ps(TWOPOW32_INV_FLOAT/2.0));
    rnd2 = _mm_fmadd_ps(rnd2, _mm_set1_ps(TWOPOW32_INV_FLOAT), _mm_set1_ps(TWOPOW32_INV_FLOAT/2.0));
    rnd3 = _mm_fmadd_ps(rnd3, _mm_set1_ps(TWOPOW32_INV_FLOAT), _mm_set1_ps(TWOPOW32_INV_FLOAT/2.0));
    rnd4 = _mm_fmadd_ps(rnd4, _mm_set1_ps(TWOPOW32_INV_FLOAT), _mm_set1_ps(TWOPOW32_INV_FLOAT/2.0));
#else
    rnd1 = _mm_mul_ps(rnd1, _mm_set1_ps(TWOPOW32_INV_FLOAT));
    rnd1 = _mm_add_ps(rnd1, _mm_set1_ps(TWOPOW32_INV_FLOAT/2.0f));
    rnd2 = _mm_mul_ps(rnd2, _mm_set1_ps(TWOPOW32_INV_FLOAT));
    rnd2 = _mm_add_ps(rnd2, _mm_set1_ps(TWOPOW32_INV_FLOAT/2.0f));
    rnd3 = _mm_mul_ps(rnd3, _mm_set1_ps(TWOPOW32_INV_FLOAT));
    rnd3 = _mm_add_ps(rnd3, _mm_set1_ps(TWOPOW32_INV_FLOAT/2.0f));
    rnd4 = _mm_mul_ps(rnd4, _mm_set1_ps(TWOPOW32_INV_FLOAT));
    rnd4 = _mm_add_ps(rnd4, _mm_set1_ps(TWOPOW32_INV_FLOAT/2.0f));
#endif
}


QUALIFIERS void aesni_double2(__m128i ctr0, __m128i ctr1, __m128i ctr2, __m128i ctr3,
                              uint32 key0, uint32 key1, uint32 key2, uint32 key3,
                              __m128d & rnd1lo, __m128d & rnd1hi, __m128d & rnd2lo, __m128d & rnd2hi)
{
    // pack input and call AES
    __m128i k128 = _mm_set_epi32(key3, key2, key1, key0);
    __m128i ctr[4] = {ctr0, ctr1, ctr2, ctr3};
    _MY_TRANSPOSE4_EPI32(ctr[0], ctr[1], ctr[2], ctr[3]);
    for (int i = 0; i < 4; ++i)
    {
        ctr[i] = aesni1xm128i(ctr[i], k128);
    }
    _MY_TRANSPOSE4_EPI32(ctr[0], ctr[1], ctr[2], ctr[3]);

    rnd1lo = _uniform_double_hq<false>(ctr[0], ctr[1]);
    rnd1hi = _uniform_double_hq<true>(ctr[0], ctr[1]);
    rnd2lo = _uniform_double_hq<false>(ctr[2], ctr[3]);
    rnd2hi = _uniform_double_hq<true>(ctr[2], ctr[3]);
}

QUALIFIERS void aesni_float4(uint32 ctr0, __m128i ctr1, uint32 ctr2, uint32 ctr3,
                             uint32 key0, uint32 key1, uint32 key2, uint32 key3,
                             __m128 & rnd1, __m128 & rnd2, __m128 & rnd3, __m128 & rnd4)
{
    __m128i ctr0v = _mm_set1_epi32(ctr0);
    __m128i ctr2v = _mm_set1_epi32(ctr2);
    __m128i ctr3v = _mm_set1_epi32(ctr3);

    aesni_float4(ctr0v, ctr1, ctr2v, ctr3v, key0, key1, key2, key3, rnd1, rnd2, rnd3, rnd4);
}

QUALIFIERS void aesni_double2(uint32 ctr0, __m128i ctr1, uint32 ctr2, uint32 ctr3,
                              uint32 key0, uint32 key1, uint32 key2, uint32 key3,
                              __m128d & rnd1lo, __m128d & rnd1hi, __m128d & rnd2lo, __m128d & rnd2hi)
{
    __m128i ctr0v = _mm_set1_epi32(ctr0);
    __m128i ctr2v = _mm_set1_epi32(ctr2);
    __m128i ctr3v = _mm_set1_epi32(ctr3);

   aesni_double2(ctr0v, ctr1, ctr2v, ctr3v, key0, key1, key2, key3, rnd1lo, rnd1hi, rnd2lo, rnd2hi);
}

QUALIFIERS void aesni_double2(uint32 ctr0, __m128i ctr1, uint32 ctr2, uint32 ctr3,
                              uint32 key0, uint32 key1, uint32 key2, uint32 key3,
                              __m128d & rnd1, __m128d & rnd2)
{
    __m128i ctr0v = _mm_set1_epi32(ctr0);
    __m128i ctr2v = _mm_set1_epi32(ctr2);
    __m128i ctr3v = _mm_set1_epi32(ctr3);

    __m128d ignore;
   aesni_double2(ctr0v, ctr1, ctr2v, ctr3v, key0, key1, key2, key3, rnd1, ignore, rnd2, ignore);
}
#endif


#ifdef __AVX2__
QUALIFIERS const std::array<__m256i,11> & aesni_roundkeys(const __m256i & k256) {
    alignas(32) std::array<uint32,8> a;
    _mm256_store_si256((__m256i*) a.data(), k256);
    
    static AlignedMap<std::array<uint32,8>, std::array<__m256i,11>> roundkeys;
    
    if(roundkeys.find(a) == roundkeys.end()) {
        auto rk1 = aesni_keygen(_mm256_extractf128_si256(k256, 0));
        auto rk2 = aesni_keygen(_mm256_extractf128_si256(k256, 1));
        for(int i = 0; i < 11; ++i) {
            roundkeys[a][i] = _my256_set_m128i(rk2[i], rk1[i]);
        }
    }
    return roundkeys[a];
}

QUALIFIERS __m256i aesni1xm128i(const __m256i & in, const __m256i & k0) {
#if defined(__VAES__)
    auto k = aesni_roundkeys(k0);
    __m256i x = _mm256_xor_si256(k[0], in);
    x = _mm256_aesenc_epi128(x, k[1]);
    x = _mm256_aesenc_epi128(x, k[2]);
    x = _mm256_aesenc_epi128(x, k[3]);
    x = _mm256_aesenc_epi128(x, k[4]);
    x = _mm256_aesenc_epi128(x, k[5]);
    x = _mm256_aesenc_epi128(x, k[6]);
    x = _mm256_aesenc_epi128(x, k[7]);
    x = _mm256_aesenc_epi128(x, k[8]);
    x = _mm256_aesenc_epi128(x, k[9]);
    x = _mm256_aesenclast_epi128(x, k[10]);
#else
    __m128i a = aesni1xm128i(_mm256_extractf128_si256(in, 0), _mm256_extractf128_si256(k0, 0));
    __m128i b = aesni1xm128i(_mm256_extractf128_si256(in, 1), _mm256_extractf128_si256(k0, 1));
    __m256i x = _my256_set_m128i(b, a);
#endif
    return x;
}

template<bool high>
QUALIFIERS __m256d _uniform_double_hq(__m256i x, __m256i y)
{
    // convert 32 to 64 bit
    if (high)
    {
        x = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(x, 1));
        y = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(y, 1));
    }
    else
    {
        x = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(x, 0));
        y = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(y, 0));
    }

    // calculate z = x ^ y << (53 - 32))
    __m256i z = _mm256_sll_epi64(y, _mm_set1_epi64x(53 - 32));
    z = _mm256_xor_si256(x, z);

    // convert uint64 to double
    __m256d rs = _my256_cvtepu64_pd(z);
    // calculate rs * TWOPOW53_INV_DOUBLE + (TWOPOW53_INV_DOUBLE/2.0)
#ifdef __FMA__
    rs = _mm256_fmadd_pd(rs, _mm256_set1_pd(TWOPOW53_INV_DOUBLE), _mm256_set1_pd(TWOPOW53_INV_DOUBLE/2.0));
#else
    rs = _mm256_mul_pd(rs, _mm256_set1_pd(TWOPOW53_INV_DOUBLE));
    rs = _mm256_add_pd(rs, _mm256_set1_pd(TWOPOW53_INV_DOUBLE/2.0));
#endif

    return rs;
}


QUALIFIERS void aesni_float4(__m256i ctr0, __m256i ctr1, __m256i ctr2, __m256i ctr3,
                              uint32 key0, uint32 key1, uint32 key2, uint32 key3,
                              __m256 & rnd1, __m256 & rnd2, __m256 & rnd3, __m256 & rnd4)
{
    // pack input and call AES
    __m256i k256 = _mm256_set_epi32(key3, key2, key1, key0, key3, key2, key1, key0);
    __m256i ctr[4] = {ctr0, ctr1, ctr2, ctr3};
    __m128i a[4], b[4];
    for (int i = 0; i < 4; ++i)
    {
        a[i] = _mm256_extractf128_si256(ctr[i], 0);
        b[i] = _mm256_extractf128_si256(ctr[i], 1);
    }
    _MY_TRANSPOSE4_EPI32(a[0], a[1], a[2], a[3]);
    _MY_TRANSPOSE4_EPI32(b[0], b[1], b[2], b[3]);
    for (int i = 0; i < 4; ++i)
    {
        ctr[i] = _my256_set_m128i(b[i], a[i]);
    }
    for (int i = 0; i < 4; ++i)
    {
        ctr[i] = aesni1xm128i(ctr[i], k256);
    }
    for (int i = 0; i < 4; ++i)
    {
        a[i] = _mm256_extractf128_si256(ctr[i], 0);
        b[i] = _mm256_extractf128_si256(ctr[i], 1);
    }
    _MY_TRANSPOSE4_EPI32(a[0], a[1], a[2], a[3]);
    _MY_TRANSPOSE4_EPI32(b[0], b[1], b[2], b[3]);
    for (int i = 0; i < 4; ++i)
    {
        ctr[i] = _my256_set_m128i(b[i], a[i]);
    }

    // convert uint32 to float
    rnd1 = _my256_cvtepu32_ps(ctr[0]);
    rnd2 = _my256_cvtepu32_ps(ctr[1]);
    rnd3 = _my256_cvtepu32_ps(ctr[2]);
    rnd4 = _my256_cvtepu32_ps(ctr[3]);
    // calculate rnd * TWOPOW32_INV_FLOAT + (TWOPOW32_INV_FLOAT/2.0f)
#ifdef __FMA__
    rnd1 = _mm256_fmadd_ps(rnd1, _mm256_set1_ps(TWOPOW32_INV_FLOAT), _mm256_set1_ps(TWOPOW32_INV_FLOAT/2.0));
    rnd2 = _mm256_fmadd_ps(rnd2, _mm256_set1_ps(TWOPOW32_INV_FLOAT), _mm256_set1_ps(TWOPOW32_INV_FLOAT/2.0));
    rnd3 = _mm256_fmadd_ps(rnd3, _mm256_set1_ps(TWOPOW32_INV_FLOAT), _mm256_set1_ps(TWOPOW32_INV_FLOAT/2.0));
    rnd4 = _mm256_fmadd_ps(rnd4, _mm256_set1_ps(TWOPOW32_INV_FLOAT), _mm256_set1_ps(TWOPOW32_INV_FLOAT/2.0));
#else
    rnd1 = _mm256_mul_ps(rnd1, _mm256_set1_ps(TWOPOW32_INV_FLOAT));
    rnd1 = _mm256_add_ps(rnd1, _mm256_set1_ps(TWOPOW32_INV_FLOAT/2.0f));
    rnd2 = _mm256_mul_ps(rnd2, _mm256_set1_ps(TWOPOW32_INV_FLOAT));
    rnd2 = _mm256_add_ps(rnd2, _mm256_set1_ps(TWOPOW32_INV_FLOAT/2.0f));
    rnd3 = _mm256_mul_ps(rnd3, _mm256_set1_ps(TWOPOW32_INV_FLOAT));
    rnd3 = _mm256_add_ps(rnd3, _mm256_set1_ps(TWOPOW32_INV_FLOAT/2.0f));
    rnd4 = _mm256_mul_ps(rnd4, _mm256_set1_ps(TWOPOW32_INV_FLOAT));
    rnd4 = _mm256_add_ps(rnd4, _mm256_set1_ps(TWOPOW32_INV_FLOAT/2.0f));
#endif
}


QUALIFIERS void aesni_double2(__m256i ctr0, __m256i ctr1, __m256i ctr2, __m256i ctr3,
                              uint32 key0, uint32 key1, uint32 key2, uint32 key3,
                              __m256d & rnd1lo, __m256d & rnd1hi, __m256d & rnd2lo, __m256d & rnd2hi)
{
    // pack input and call AES
    __m256i k256 = _mm256_set_epi32(key3, key2, key1, key0, key3, key2, key1, key0);
    __m256i ctr[4] = {ctr0, ctr1, ctr2, ctr3};
    __m128i a[4], b[4];
    for (int i = 0; i < 4; ++i)
    {
        a[i] = _mm256_extractf128_si256(ctr[i], 0);
        b[i] = _mm256_extractf128_si256(ctr[i], 1);
    }
    _MY_TRANSPOSE4_EPI32(a[0], a[1], a[2], a[3]);
    _MY_TRANSPOSE4_EPI32(b[0], b[1], b[2], b[3]);
    for (int i = 0; i < 4; ++i)
    {
        ctr[i] = _my256_set_m128i(b[i], a[i]);
    }
    for (int i = 0; i < 4; ++i)
    {
        ctr[i] = aesni1xm128i(ctr[i], k256);
    }
    for (int i = 0; i < 4; ++i)
    {
        a[i] = _mm256_extractf128_si256(ctr[i], 0);
        b[i] = _mm256_extractf128_si256(ctr[i], 1);
    }
    _MY_TRANSPOSE4_EPI32(a[0], a[1], a[2], a[3]);
    _MY_TRANSPOSE4_EPI32(b[0], b[1], b[2], b[3]);
    for (int i = 0; i < 4; ++i)
    {
        ctr[i] = _my256_set_m128i(b[i], a[i]);
    }

    rnd1lo = _uniform_double_hq<false>(ctr[0], ctr[1]);
    rnd1hi = _uniform_double_hq<true>(ctr[0], ctr[1]);
    rnd2lo = _uniform_double_hq<false>(ctr[2], ctr[3]);
    rnd2hi = _uniform_double_hq<true>(ctr[2], ctr[3]);
}

QUALIFIERS void aesni_float4(uint32 ctr0, __m256i ctr1, uint32 ctr2, uint32 ctr3,
                             uint32 key0, uint32 key1, uint32 key2, uint32 key3,
                             __m256 & rnd1, __m256 & rnd2, __m256 & rnd3, __m256 & rnd4)
{
    __m256i ctr0v = _mm256_set1_epi32(ctr0);
    __m256i ctr2v = _mm256_set1_epi32(ctr2);
    __m256i ctr3v = _mm256_set1_epi32(ctr3);

    aesni_float4(ctr0v, ctr1, ctr2v, ctr3v, key0, key1, key2, key3, rnd1, rnd2, rnd3, rnd4);
}

QUALIFIERS void aesni_double2(uint32 ctr0, __m256i ctr1, uint32 ctr2, uint32 ctr3,
                              uint32 key0, uint32 key1, uint32 key2, uint32 key3,
                              __m256d & rnd1lo, __m256d & rnd1hi, __m256d & rnd2lo, __m256d & rnd2hi)
{
    __m256i ctr0v = _mm256_set1_epi32(ctr0);
    __m256i ctr2v = _mm256_set1_epi32(ctr2);
    __m256i ctr3v = _mm256_set1_epi32(ctr3);

    aesni_double2(ctr0v, ctr1, ctr2v, ctr3v, key0, key1, key2, key3, rnd1lo, rnd1hi, rnd2lo, rnd2hi);
}

QUALIFIERS void aesni_double2(uint32 ctr0, __m256i ctr1, uint32 ctr2, uint32 ctr3,
                              uint32 key0, uint32 key1, uint32 key2, uint32 key3,
                              __m256d & rnd1, __m256d & rnd2)
{
#if 0
    __m256i ctr0v = _mm256_set1_epi32(ctr0);
    __m256i ctr2v = _mm256_set1_epi32(ctr2);
    __m256i ctr3v = _mm256_set1_epi32(ctr3);

    __m256d ignore;
    aesni_double2(ctr0v, ctr1, ctr2v, ctr3v, key0, key1, key2, key3, rnd1, ignore, rnd2, ignore);
#else
    __m128d rnd1lo, rnd1hi, rnd2lo, rnd2hi;
    aesni_double2(ctr0, _mm256_extractf128_si256(ctr1, 0), ctr2, ctr3, key0, key1, key2, key3, rnd1lo, rnd1hi, rnd2lo, rnd2hi);
    rnd1 = _my256_set_m128d(rnd1hi, rnd1lo);
    rnd2 = _my256_set_m128d(rnd2hi, rnd2lo);
#endif
}
#endif


#ifdef __AVX512F__
QUALIFIERS const std::array<__m512i,11> & aesni_roundkeys(const __m512i & k512) {
    alignas(64) std::array<uint32,16> a;
    _mm512_store_si512((__m512i*) a.data(), k512);
    
    static AlignedMap<std::array<uint32,16>, std::array<__m512i,11>> roundkeys;
    
    if(roundkeys.find(a) == roundkeys.end()) {
        auto rk1 = aesni_keygen(_mm512_extracti32x4_epi32(k512, 0));
        auto rk2 = aesni_keygen(_mm512_extracti32x4_epi32(k512, 1));
        auto rk3 = aesni_keygen(_mm512_extracti32x4_epi32(k512, 2));
        auto rk4 = aesni_keygen(_mm512_extracti32x4_epi32(k512, 3));
        for(int i = 0; i < 11; ++i) {
            roundkeys[a][i] = _my512_set_m128i(rk4[i], rk3[i], rk2[i], rk1[i]);
        }
    }
    return roundkeys[a];
}

QUALIFIERS __m512i aesni1xm128i(const __m512i & in, const __m512i & k0) {
#ifdef __VAES__
    auto k = aesni_roundkeys(k0);
    __m512i x = _mm512_xor_si512(k[0], in);
    x = _mm512_aesenc_epi128(x, k[1]);
    x = _mm512_aesenc_epi128(x, k[2]);
    x = _mm512_aesenc_epi128(x, k[3]);
    x = _mm512_aesenc_epi128(x, k[4]);
    x = _mm512_aesenc_epi128(x, k[5]);
    x = _mm512_aesenc_epi128(x, k[6]);
    x = _mm512_aesenc_epi128(x, k[7]);
    x = _mm512_aesenc_epi128(x, k[8]);
    x = _mm512_aesenc_epi128(x, k[9]);
    x = _mm512_aesenclast_epi128(x, k[10]);
#else
    __m128i a = aesni1xm128i(_mm512_extracti32x4_epi32(in, 0), _mm512_extracti32x4_epi32(k0, 0));
    __m128i b = aesni1xm128i(_mm512_extracti32x4_epi32(in, 1), _mm512_extracti32x4_epi32(k0, 1));
    __m128i c = aesni1xm128i(_mm512_extracti32x4_epi32(in, 2), _mm512_extracti32x4_epi32(k0, 2));
    __m128i d = aesni1xm128i(_mm512_extracti32x4_epi32(in, 3), _mm512_extracti32x4_epi32(k0, 3));
    __m512i x = _my512_set_m128i(d, c, b, a);
#endif
    return x;
}

template<bool high>
QUALIFIERS __m512d _uniform_double_hq(__m512i x, __m512i y)
{
    // convert 32 to 64 bit
    if (high)
    {
        x = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(x, 1));
        y = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(y, 1));
    }
    else
    {
        x = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(x, 0));
        y = _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(y, 0));
    }

    // calculate z = x ^ y << (53 - 32))
    __m512i z = _mm512_sll_epi64(y, _mm_set1_epi64x(53 - 32));
    z = _mm512_xor_si512(x, z);

    // convert uint64 to double
    __m512d rs = _mm512_cvtepu64_pd(z);
    // calculate rs * TWOPOW53_INV_DOUBLE + (TWOPOW53_INV_DOUBLE/2.0)
    rs = _mm512_fmadd_pd(rs, _mm512_set1_pd(TWOPOW53_INV_DOUBLE), _mm512_set1_pd(TWOPOW53_INV_DOUBLE/2.0));

    return rs;
}


QUALIFIERS void aesni_float4(__m512i ctr0, __m512i ctr1, __m512i ctr2, __m512i ctr3,
                              uint32 key0, uint32 key1, uint32 key2, uint32 key3,
                              __m512 & rnd1, __m512 & rnd2, __m512 & rnd3, __m512 & rnd4)
{
    // pack input and call AES
    __m512i k512 = _mm512_set_epi32(key3, key2, key1, key0, key3, key2, key1, key0,
                                    key3, key2, key1, key0, key3, key2, key1, key0);
    __m512i ctr[4] = {ctr0, ctr1, ctr2, ctr3};
    __m128i a[4], b[4], c[4], d[4];
    for (int i = 0; i < 4; ++i)
    {
        a[i] = _mm512_extracti32x4_epi32(ctr[i], 0);
        b[i] = _mm512_extracti32x4_epi32(ctr[i], 1);
        c[i] = _mm512_extracti32x4_epi32(ctr[i], 2);
        d[i] = _mm512_extracti32x4_epi32(ctr[i], 3);
    }
    _MY_TRANSPOSE4_EPI32(a[0], a[1], a[2], a[3]);
    _MY_TRANSPOSE4_EPI32(b[0], b[1], b[2], b[3]);
    _MY_TRANSPOSE4_EPI32(c[0], c[1], c[2], c[3]);
    _MY_TRANSPOSE4_EPI32(d[0], d[1], d[2], d[3]);
    for (int i = 0; i < 4; ++i)
    {
        ctr[i] = _my512_set_m128i(d[i], c[i], b[i], a[i]);
    }
    for (int i = 0; i < 4; ++i)
    {
        ctr[i] = aesni1xm128i(ctr[i], k512);
    }
    for (int i = 0; i < 4; ++i)
    {
        a[i] = _mm512_extracti32x4_epi32(ctr[i], 0);
        b[i] = _mm512_extracti32x4_epi32(ctr[i], 1);
        c[i] = _mm512_extracti32x4_epi32(ctr[i], 2);
        d[i] = _mm512_extracti32x4_epi32(ctr[i], 3);
    }
    _MY_TRANSPOSE4_EPI32(a[0], a[1], a[2], a[3]);
    _MY_TRANSPOSE4_EPI32(b[0], b[1], b[2], b[3]);
    _MY_TRANSPOSE4_EPI32(c[0], c[1], c[2], c[3]);
    _MY_TRANSPOSE4_EPI32(d[0], d[1], d[2], d[3]);
    for (int i = 0; i < 4; ++i)
    {
        ctr[i] = _my512_set_m128i(d[i], c[i], b[i], a[i]);
    }

    // convert uint32 to float
    rnd1 = _mm512_cvtepu32_ps(ctr[0]);
    rnd2 = _mm512_cvtepu32_ps(ctr[1]);
    rnd3 = _mm512_cvtepu32_ps(ctr[2]);
    rnd4 = _mm512_cvtepu32_ps(ctr[3]);
    // calculate rnd * TWOPOW32_INV_FLOAT + (TWOPOW32_INV_FLOAT/2.0f)
    rnd1 = _mm512_fmadd_ps(rnd1, _mm512_set1_ps(TWOPOW32_INV_FLOAT), _mm512_set1_ps(TWOPOW32_INV_FLOAT/2.0));
    rnd2 = _mm512_fmadd_ps(rnd2, _mm512_set1_ps(TWOPOW32_INV_FLOAT), _mm512_set1_ps(TWOPOW32_INV_FLOAT/2.0));
    rnd3 = _mm512_fmadd_ps(rnd3, _mm512_set1_ps(TWOPOW32_INV_FLOAT), _mm512_set1_ps(TWOPOW32_INV_FLOAT/2.0));
    rnd4 = _mm512_fmadd_ps(rnd4, _mm512_set1_ps(TWOPOW32_INV_FLOAT), _mm512_set1_ps(TWOPOW32_INV_FLOAT/2.0));
}


QUALIFIERS void aesni_double2(__m512i ctr0, __m512i ctr1, __m512i ctr2, __m512i ctr3,
                              uint32 key0, uint32 key1, uint32 key2, uint32 key3,
                              __m512d & rnd1lo, __m512d & rnd1hi, __m512d & rnd2lo, __m512d & rnd2hi)
{
    // pack input and call AES
    __m512i k512 = _mm512_set_epi32(key3, key2, key1, key0, key3, key2, key1, key0,
                                    key3, key2, key1, key0, key3, key2, key1, key0);
    __m512i ctr[4] = {ctr0, ctr1, ctr2, ctr3};
    __m128i a[4], b[4], c[4], d[4];
    for (int i = 0; i < 4; ++i)
    {
        a[i] = _mm512_extracti32x4_epi32(ctr[i], 0);
        b[i] = _mm512_extracti32x4_epi32(ctr[i], 1);
        c[i] = _mm512_extracti32x4_epi32(ctr[i], 2);
        d[i] = _mm512_extracti32x4_epi32(ctr[i], 3);
    }
    _MY_TRANSPOSE4_EPI32(a[0], a[1], a[2], a[3]);
    _MY_TRANSPOSE4_EPI32(b[0], b[1], b[2], b[3]);
    _MY_TRANSPOSE4_EPI32(c[0], c[1], c[2], c[3]);
    _MY_TRANSPOSE4_EPI32(d[0], d[1], d[2], d[3]);
    for (int i = 0; i < 4; ++i)
    {
        ctr[i] = _my512_set_m128i(d[i], c[i], b[i], a[i]);
    }
    for (int i = 0; i < 4; ++i)
    {
        ctr[i] = aesni1xm128i(ctr[i], k512);
    }
    for (int i = 0; i < 4; ++i)
    {
        a[i] = _mm512_extracti32x4_epi32(ctr[i], 0);
        b[i] = _mm512_extracti32x4_epi32(ctr[i], 1);
        c[i] = _mm512_extracti32x4_epi32(ctr[i], 2);
        d[i] = _mm512_extracti32x4_epi32(ctr[i], 3);
    }
    _MY_TRANSPOSE4_EPI32(a[0], a[1], a[2], a[3]);
    _MY_TRANSPOSE4_EPI32(b[0], b[1], b[2], b[3]);
    _MY_TRANSPOSE4_EPI32(c[0], c[1], c[2], c[3]);
    _MY_TRANSPOSE4_EPI32(d[0], d[1], d[2], d[3]);
    for (int i = 0; i < 4; ++i)
    {
        ctr[i] = _my512_set_m128i(d[i], c[i], b[i], a[i]);
    }

    rnd1lo = _uniform_double_hq<false>(ctr[0], ctr[1]);
    rnd1hi = _uniform_double_hq<true>(ctr[0], ctr[1]);
    rnd2lo = _uniform_double_hq<false>(ctr[2], ctr[3]);
    rnd2hi = _uniform_double_hq<true>(ctr[2], ctr[3]);
}

QUALIFIERS void aesni_float4(uint32 ctr0, __m512i ctr1, uint32 ctr2, uint32 ctr3,
                             uint32 key0, uint32 key1, uint32 key2, uint32 key3,
                             __m512 & rnd1, __m512 & rnd2, __m512 & rnd3, __m512 & rnd4)
{
    __m512i ctr0v = _mm512_set1_epi32(ctr0);
    __m512i ctr2v = _mm512_set1_epi32(ctr2);
    __m512i ctr3v = _mm512_set1_epi32(ctr3);

    aesni_float4(ctr0v, ctr1, ctr2v, ctr3v, key0, key1, key2, key3, rnd1, rnd2, rnd3, rnd4);
}

QUALIFIERS void aesni_double2(uint32 ctr0, __m512i ctr1, uint32 ctr2, uint32 ctr3,
                              uint32 key0, uint32 key1, uint32 key2, uint32 key3,
                              __m512d & rnd1lo, __m512d & rnd1hi, __m512d & rnd2lo, __m512d & rnd2hi)
{
    __m512i ctr0v = _mm512_set1_epi32(ctr0);
    __m512i ctr2v = _mm512_set1_epi32(ctr2);
    __m512i ctr3v = _mm512_set1_epi32(ctr3);

    aesni_double2(ctr0v, ctr1, ctr2v, ctr3v, key0, key1, key2, key3, rnd1lo, rnd1hi, rnd2lo, rnd2hi);
}

QUALIFIERS void aesni_double2(uint32 ctr0, __m512i ctr1, uint32 ctr2, uint32 ctr3,
                              uint32 key0, uint32 key1, uint32 key2, uint32 key3,
                              __m512d & rnd1, __m512d & rnd2)
{
#if 0
    __m512i ctr0v = _mm512_set1_epi32(ctr0);
    __m512i ctr2v = _mm512_set1_epi32(ctr2);
    __m512i ctr3v = _mm512_set1_epi32(ctr3);

    __m512d ignore;
    aesni_double2(ctr0v, ctr1, ctr2v, ctr3v, key0, key1, key2, key3, rnd1, ignore, rnd2, ignore);
#else
   __m256d rnd1lo, rnd1hi, rnd2lo, rnd2hi;
   aesni_double2(ctr0, _mm512_extracti64x4_epi64(ctr1, 0), ctr2, ctr3, key0, key1, key2, key3, rnd1lo, rnd1hi, rnd2lo, rnd2hi);
   rnd1 = _my512_set_m256d(rnd1hi, rnd1lo);
   rnd2 = _my512_set_m256d(rnd2hi, rnd2lo);
#endif
}
#endif

