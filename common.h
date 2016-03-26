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
#include <memory>

#ifndef EMULATE
#include <immintrin.h>
#else
#include <avxintrin-emu.h>
#endif

static const size_t k_vecsize = 8;

//32 byte alignment for avx2. 64 in case of avxknc
//also makes false sharing harder, bc line is all for this.
static const size_t k_align = 64;

template <typename T> auto allocate_aligned(int size){

	auto tmp1 = size*sizeof(T);
	auto sz = tmp1 + k_align - (tmp1 % k_align);
	auto ptr = (T*)_mm_malloc(sz, k_align);
  auto del = [](auto *p){ _mm_free(p); };

  return std::unique_ptr<T, decltype(del)>(ptr, del);
}

template <typename T> using aligned_ptr =  decltype(allocate_aligned<T>(0));

inline uint32_t MarsagliaXOR(uint32_t *p_seed) {
    auto seed = *p_seed;

    if (seed == 0) {
        seed = 1;
    }

    seed ^= seed >> 5;
    seed ^= seed << 21;
    seed ^= seed >> 7;

    *p_seed = seed;
    return seed;
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
