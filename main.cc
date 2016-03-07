#include <benchmark/benchmark.h>
#include <tuple>
#include <immintrin.h>
//#include <tbb>
#include <stdlib.h>
#include "common.h"

const int k_max = 128;
const int middle = k_max >> 1;
const int quarter = middle >> 1;
const int threeq = middle + quarter;

void init_data(int *d, int len)
{
  unsigned int seed = 0;
  for (int i =0 ; i < len; ++i){
    d[i] = rand_r(&seed) % k_max;
  }
}

tuple<int,int,int> countA(int *d, int len, int lim1, int lim2) {
  long int counter1 = 0;
  long int counter2 = 0;

  /**
   * WTS: beats vectorized version for some data distributions.
   */
  for (int i =0; i < len; ++i) {
      if (d[i] > lim1) {
	counter1++;

	// only does work for qualifying tuples 
	if (d[i] > lim2) {
	  counter2++;
	}
      }
  }

  return make_tuple(len - counter1, counter2 - counter1, counter2);
}

tuple<int,int,int> countB(int *d, int len, int lim1, int lim2) {
  __m256i counter1_ = _mm256_set1_epi32(0);
  __m256i counter2_ = _mm256_set1_epi32(0);
  
  const __m256i lim1_ = _mm256_set1_epi32(lim1);
  const __m256i lim2_ = _mm256_set1_epi32(lim2);

  for (int i = 0; i < len; i+=sizeof(__m256i)) {
    auto data_ = _mm256_load_si256((__m256i*)&(d[i]));
    {
      auto tmpgt1_ = _mm256_cmpgt_epi32(data_, lim1_); // gt returns 0 or 0xffff..f
      auto count1_ = _mm256_srli_epi32(tmpgt1_, 31); // only keep largest bit
      counter1_ = _mm256_add_epi32(count1_, counter1_); // increment
    }

    { // TODO: and qualifying tuples here with the previous flags, because we want to simulate predication (while controlling number of entries that qualify), even though in this case one implies the other.
      auto tmpgt2_ = _mm256_cmpgt_epi32(data_, lim2_);
      auto count2_ = _mm256_srli_epi32(tmpgt2_, 31); // only keep largest bit
      counter2_ = _mm256_add_epi32(count2_, counter2_); // increment
    }
  }

  auto counter1 = sum_lanes_8(counter1_);
  auto counter2 = sum_lanes_8(counter2_);

  return make_tuple(len - counter1, counter2 - counter1, counter2);
}

template <typename Func> void BM_splitvec(benchmark::State & state, Func f){
  int * data = new int[state.range_x()] {};
  init_data(data, state.range_x()); 

  while (state.KeepRunning()) {
    f(data, state.range_y(), state.range_x(), state.range_x() + 32);
  }  
}

void BM_plain(benchmark::State & state) {
  BM_splitvec(state, countA);
}

void BM_fullvec(benchmark::State & state) {
  BM_splitvec(state, countB);
}

BENCHMARK(BM_plain)->Arg();
BENCHMARK(BM_fullvec);

BENCHMARK_MAIN();
