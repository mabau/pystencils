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

inline void cachelineZero(void * p) {
	__asm__ volatile("dc zva, %0"::"r"(p));
}

inline size_t _cachelineSize() {
	// check that dc zva is permitted
	uint64_t dczid;
	__asm__ volatile ("mrs %0, dczid_el0" : "=r"(dczid));
	if ((dczid & (1 << 4)) != 0) {
		return SIZE_MAX;
	}

	// allocate and fill with ones
	const size_t max_size = 0x100000;
	uint8_t data[2*max_size];
	for (size_t i = 0; i < 2*max_size; ++i) {
		data[i] = 0xff;
	}
	
	// find alignment offset
	size_t offset = max_size - ((uintptr_t) data) % max_size;

	// zero a cacheline
	cachelineZero((void*) (data + offset));

	// make sure that at least one byte was zeroed
	if (data[offset] != 0) {
		return SIZE_MAX;
	}

	// make sure that nothing was zeroed before the pointer
	if (data[offset-1] == 0) {
		return SIZE_MAX;
	}

	// find the last byte that was zeroed
	for (size_t size = 1; size < max_size; ++size) {
		if (data[offset + size] != 0) {
			return size;
		}
	}
	
	// too much was zeroed
	return SIZE_MAX;
}

inline size_t cachelineSize() {
	static size_t size = _cachelineSize();
	return size;
}
