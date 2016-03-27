#include "impl.h"
#include <vectorclass/vectorclass.h>
#include <iostream>
#include <iomanip>

#define SALIGN __declspec (align(64))

static_assert(VECTORI256_H == 2, "use int 256 bit vectors");
using namespace std;

template <typename T> struct vec {
	class t;
};

template <> struct vec<int8_t> {
	typedef Vec32c t;
};

template <> struct vec<int16_t> {
	typedef Vec16s t;
};

template <> struct vec<int32_t> {
	typedef Vec8i t;
};

template <> struct vec<int64_t> {
	typedef Vec4q t;
};

typedef vec<data_t>::t vec_t;

const static size_t k_buf_size = 2048;
const static size_t k_vec_size = sizeof(vec_t);
static_assert(k_buf_size % k_vec_size == 0, "vector must divide buffer");
static_assert(k_vec_size % sizeof(data_t) == 0, "data_t must divide vector");

const static size_t k_elts_per_vec = k_vec_size/sizeof(data_t);
const static size_t k_elts_per_buf = k_buf_size/sizeof(data_t);

template <typename T>  typename vec<T>::t gather(uint32_t const * index, T * table);


template <> typename vec<int8_t>::t gather<int8_t>(uint32_t const * index, int8_t * table){
	__m256i tmp1 = _mm256_load_si256((__m256i const *)index);
	auto grp1 =  _mm256_i32gather_epi32((int32_t*)table, tmp1, sizeof(int16_t));
	__m256i tmp2 = _mm256_load_si256(((__m256i const *)index) + 1 );
	auto grp2 =  _mm256_i32gather_epi32((int32_t*)table, tmp2, sizeof(int16_t));

	// will reorder things, but thats okay since it reorders every column equally
	auto merged = _mm256_unpacklo_epi16(grp1, grp2);
	return merged;
}

template <> typename vec<int16_t>::t gather<int16_t>(uint32_t const * index, int16_t * table)
{
	__m256i tmp1 = _mm256_load_si256((__m256i const *)index);
	__m256i grp1 =  _mm256_i32gather_epi32((int32_t*)table, tmp1, sizeof(int16_t));
	// every second short here is not valid data
	
	__m256i tmp2 = _mm256_load_si256(((__m256i const *)index) + 1 );
	__m256i grp2 =  _mm256_i32gather_epi32((int32_t*)table, tmp2, sizeof(int16_t));

	auto grp1b = _mm256_unpacklo_epi16(grp1, grp2);
	auto grp2b = _mm256_unpackhi_epi16(grp1, grp2);

	grp1 = _mm256_unpacklo_epi16(grp1b, grp2b);
	grp2 = _mm256_unpackhi_epi16(grp1b, grp2b);

	grp1b = _mm256_unpacklo_epi16(grp1, grp2);
	// grp1b has the correct solution, but in the wrong order.
	// in blocks of 64 bits, this is what it looks like a[0] a[2] a[1] a[3]
	// mask must be 0213:    11 01 10 00
	auto final = _mm256_permute4x64_epi64(grp1b, 0b11'01'10'00);
	return final;
}


template <> typename vec<int32_t>::t gather<int32_t>(uint32_t const * index, int32_t * table){
	__m256i tmp1 = _mm256_load_si256((__m256i const *)index);
	return _mm256_i32gather_epi32(table, tmp1, sizeof(int32_t));
}

template <> typename vec<int64_t>::t gather<int64_t>(uint32_t const * index, int64_t * table){
	
	constexpr uint64_t mask32 = ~((1ULL << 5) - 1); // 0xfffffff..000 mask out last 4 bits 
	auto aligned32 = (uint32_t*)((uint64_t)index & mask32); //expect last few bits 0.
	auto pick_high_half = (int32_t)(index == aligned32) - 1; // expect 0xfffff..f if true,0x0000..00 otherwise. 
	//printf("0x%x ",pick_high_half);
	__m256i tmp1 = _mm256_load_si256((__m256i const*)aligned32);

	// we don't know ahead of time which half it is we want,
	// but the intrinsic requires a compile time constant....
	Vec4i lower(_mm256_extracti128_si256(tmp1, 0));
  Vec4i higher(_mm256_extracti128_si256(tmp1, 1));
	Vec4i mask(pick_high_half);
	auto indices = (lower & ~mask) | (higher & mask);

	return _mm256_i32gather_epi64(table, indices, sizeof(int64_t));
}


inline ostream& operator<<(ostream &o, const Vec16s & v){
	o << "Vec16s ";
	for (int i = 0; i < 16; ++i){
		o << std::setfill('0') << std::setw(2) << v[i] << ",";
	}
	o << endl;
	return o;
}


inline ostream& operator<<(ostream &o, const Vec8i & v){
	using namespace std;
	o << "Vec8i ";
	for (int i = 0; i < 8; ++i){
		o << setfill('0') << setw(2) << v[i] << ",";
	}
	o << endl;
	return o;
}


inline ostream& operator<<(ostream &o, const Vec4q & v){
	using namespace std;
	o << "Vec4q ";
	for (int i = 0; i < 4; ++i){
		o << setfill('0') << setw(2) << v[i] << ",";
	}
	o << endl;
	return o;
}

