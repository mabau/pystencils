#include <altivec.h>
#undef vector
#undef bool

inline void cachelineZero(void * p) {
#ifdef __xlC__
	__dcbz(p);
#else
	__asm__ volatile("dcbz 0, %0"::"r"(p):"memory");
#endif
}

inline size_t _cachelineSize() {
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
