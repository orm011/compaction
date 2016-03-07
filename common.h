/*
 * common.h
 *
 *  Created on: Sep 30, 2015
 *      Author: orm
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <cstring>
#include <string>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <cassert>
#include <thread>
#include <utility>
#include <unordered_map>

#ifndef EMULATE
#include <immintrin.h>
#else
#include <avxintrin-emu.h>
#endif

inline int MarsagliaXOR(int *p_seed) {
    int seed = *p_seed;

    if (seed == 0) {
        seed = 1;
    }

    seed ^= seed << 6;
    seed ^= ((unsigned)seed) >> 21;
    seed ^= seed << 7;

    *p_seed = seed;

    return seed & 0x7FFFFFFF;
}

template <typename T> T* allocate(size_t len, size_t byte_alignment = 64) {
	 char *p = (char*)malloc(sizeof(T)*len + byte_alignment); // extra padding

	 if (byte_alignment > 0  && ((size_t)p) % byte_alignment != 0) {
		 p += byte_alignment;
		 auto remainder = ((size_t)p) % byte_alignment;
		 p -= remainder;
		 assert (((size_t)p % byte_alignment) == 0);
	 }

	 return (T*)p;
}

#define as_array(r) ((int32_t*)(&(r)))
static const size_t k_vecsize = 8;
static const __m256i _minus1 = _mm256_set1_epi32(0xffffffff);
static const __m256i _ones = _mm256_set1_epi32(1);

inline int64_t sum_lanes_8(const __m256i & vector){
  (void)_ones;
  (void)_minus1;
  
	int64_t total = 0;
	const int32_t *p = (int32_t*)&vector;
	for (int lane = 0; lane < 8; ++lane) {
		total += p[lane];
	}
	return total;
}

#endif /* COMMON_H_ */
