#include "impl.h"
#include <vectorclass/vectorclass.h>
#include <iostream>
#include <iomanip>
#include "mask_table.h"

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
static const size_t k_cache_line_size = 64;
const static size_t k_elts_per_line = k_cache_line_size / sizeof(data_t);

static const size_t k_acc_size = sizeof(int64_t)/sizeof(data_t);

template <size_t N> struct Vec4qVec {
	SALIGN Vec4q arr[N] {};

	Vec4qVec() = default;
	
	Vec4qVec(int64_t init){
		auto init_vec = Vec4q(init);
		for (int i = 0; i < N; ++i) {
			arr[i].load(&init_vec);
		}
	}

	inline Vec4qVec & operator=(const Vec4qVec &v)
	{
		for (int i = 0; i < N; ++i){
			arr[i] = v.arr[i];
		}
	}

	inline Vec4qVec & operator+=(const Vec4qVec<N> &right)
		{
			for (int i = 0; i < N; ++i) {
				arr[i] += right.arr[i];
			}

			return *this;
		}

	template <size_t S, size_t M> void assign_at(const Vec4qVec<M> &src){
		static_assert(S + M <= N, "copy would overflow");
		for (int i = S; i < S + M; ++i){
			arr[i].load_a(&src.arr[i - S]);
		}
	}

	int64_t operator[](size_t i){
		return arr[i/4][i%4];
	}


	inline int64_t sum_all(){
		int64_t total = 0;
		for (int i =0 ; i < N; ++i){
			total+= horizontal_add(arr[i]);
		}

		return total;
	}
};

inline Vec4qVec<1> extend(Vec4q quads){
	Vec4qVec<1> ans;
	ans.arr[0] = quads;
	return ans;
}

inline Vec4qVec<2> extend(Vec8i ints){
	Vec4qVec<2> ans;
	
	Vec4q tmp1 = extend_low(ints);
	Vec4q tmp2 = extend_high(ints);

 	Vec4qVec<1> low = extend(tmp1);
	Vec4qVec<1> high = extend(tmp2);
	
	ans.assign_at<0>(low);
	ans.assign_at<1>(high);
	return ans;
}

inline Vec4qVec<4> extend(Vec16s shorts){
	Vec4qVec<4> ans;
	Vec8i tmp1 = extend_low(shorts);
	Vec8i tmp2 = extend_high(shorts);

	Vec4qVec<2> low = extend(tmp1);
	Vec4qVec<2> high = extend(tmp2);

	ans.assign_at<0>(low);
	ans.assign_at<2>(high);
	
	return ans;	
}

inline Vec4qVec<8> extend(Vec32c chars){
 	Vec4qVec<8> ans(0);

	Vec16s tmp1 = extend_low(chars);
	Vec16s tmp2 = extend_high(chars);

	Vec4qVec<4> low = extend(tmp1);
	Vec4qVec<4> high = extend(tmp2);

	ans.assign_at<0>(low);
	ans.assign_at<4>(high);
	return ans;	
}


static_assert(((k_elts_per_buf - 1) & k_elts_per_buf) == 0, "elts per buf should be power of 2");

template <typename T>  typename vec<T>::t gather(uint32_t const * index, T * table);

template <typename T>  void buffer_addresses(uint32_t *, const T *, int *, uint32_t *, const q19params &);

template <> void buffer_addresses<int8_t>(uint32_t *iptr, const int8_t * startbrand, int *jptr, uint32_t *buf, const q19params &p1)
{
	auto &i = *iptr;
	auto &j = *jptr;
	
	vec_t currbrands;
	currbrands.load_a(&startbrand[i]);
	auto quals = currbrands == p1.brand;
	auto bigmask = _mm256_movemask_epi8(quals);
	
	for (int sub = 0; sub < 4; ++sub) {
		auto charmask = (bigmask >> sub*8) & 0xff; // pick the lowest 8 bits
		auto delta_j = _mm_popcnt_u64(charmask);
		Vec8ui perm_mask;
		perm_mask.load_a(&mask_table[charmask]);
		auto offset = i + sub*8;
		perm_mask += offset;
		_mm256_storeu_si256((__m256i*)&buf[j], perm_mask);
		j+=delta_j;
	}
}

template <> void buffer_addresses<int32_t>(uint32_t *iptr, const int32_t * startbrand, int *jptr, uint32_t *buf, const q19params &p1)
{
	auto &i = *iptr;
	auto &j = *jptr;
	vec_t currbrands;
	currbrands.load_a(&startbrand[i]);
	auto quals = currbrands == p1.brand;
	auto charmask = _mm256_movemask_ps(_mm256_castsi256_ps(quals));
	auto delta_j = _mm_popcnt_u64(charmask);
	vec_t perm_mask;
	perm_mask.load_a(&mask_table[charmask]);
	auto store_mask = _mm256_permutevar8x32_epi32(quals, perm_mask);
	auto store_pos = perm_mask + i;
	_mm256_maskstore_epi32((int*)&buf[j], store_mask, store_pos);
	j+=delta_j;
}


template <> typename vec<int8_t>::t gather<int8_t>(uint32_t const * index, int8_t * table){

	// all 32 next positions 
	__m256i tmp1 = _mm256_load_si256((__m256i const *)index);
	auto grp1 =  _mm256_i32gather_epi32((int32_t*)table, tmp1, 1);
	__m256i tmp2 = _mm256_load_si256(((__m256i const *)index) + 1);
	auto grp2 =  _mm256_i32gather_epi32((int32_t*)table, tmp2, 1);
	__m256i tmp3 = _mm256_load_si256(((__m256i const *)index) + 2);
	auto grp3 =  _mm256_i32gather_epi32((int32_t*)table, tmp3, 1);
	__m256i tmp4 = _mm256_load_si256(((__m256i const *)index) + 3);
	auto grp4 =  _mm256_i32gather_epi32((int32_t*)table, tmp4, 1);
	
	// will reorder things, but thats okay since it reorders every column equally
	auto collect_valid =
		_mm256_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
										 0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

	// all 4 valid chars (per 128 bit lane) get grouped on the low side of the lane.
	auto grp1left = _mm256_shuffle_epi8(grp1, collect_valid);
	auto grp2left = _mm256_shuffle_epi8(grp2, collect_valid);
	auto grp3left = _mm256_shuffle_epi8(grp3, collect_valid);
	auto grp4left = _mm256_shuffle_epi8(grp4, collect_valid);

	// lower 8 chars of each lane get interleaved. the first 8 chars of each lane
	// are valid now
	auto grp12merged = _mm256_unpacklo_epi32(grp1left, grp2left);
	auto grp34merged = _mm256_unpacklo_epi32(grp3left, grp4left);

	auto lower = _mm256_unpacklo_epi64(grp12merged, grp34merged); // slightly reordered
	return lower;
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


inline ostream& operator<<(ostream &o, const Vec32c & v){
	o << "Vec32c ";
	for (int i = 0; i < 32; ++i) {
		o << std::setfill('0') << std::setw(2) << (uint32_t)v[i] << ",";
	}
	o << endl;
	return o;
}

inline ostream& operator<<(ostream &o, const Vec16s & v){
	o << "Vec16s ";
	for (int i = 0; i < 16; ++i){
		o << std::setfill('0') << std::setw(2) << (uint32_t)v[i] << ",";
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

