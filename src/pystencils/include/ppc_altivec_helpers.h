/*
Copyright 2021, Michael Kuron.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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
