#include "common.h"
#include "impl.h"
#include <tbb/tbb.h>


using namespace std;

tuple<int,int,int> count_naive(int *d, int len, int lim1, int lim2) {
  int counter1 = 0;
  int counter2 = 0;

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

  return make_tuple(len - counter1, counter1 - counter2, counter2);
}

tuple<int,int,int> count_mask(int *d, int len, int lim1, int lim2) {
  __m256i counter1_ = _mm256_set1_epi32(0);
  __m256i counter2_ = _mm256_set1_epi32(0);
  
  const __m256i lim1_ = _mm256_set1_epi32(lim1);
  const __m256i lim2_ = _mm256_set1_epi32(lim2);

  for (int i = 0; i < len; i+=k_vecsize) {
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

  return make_tuple(len - counter1, counter1 - counter2, counter2);
}



tuple<int,int,int> count_mask_2unroll(int *d, int len, int lim1, int lim2) {
  __m256i counter11_ = _mm256_set1_epi32(0);
  __m256i counter21_ = _mm256_set1_epi32(0);
  __m256i counter12_ = _mm256_set1_epi32(0);
  __m256i counter22_ = _mm256_set1_epi32(0);

  const __m256i lim1_ = _mm256_set1_epi32(lim1);
  const __m256i lim2_ = _mm256_set1_epi32(lim2);

  int i = 0;
  const auto stride = k_vecsize << 1;
  for (i = 0; i < len; i+=stride) {
    {
      auto data_ = _mm256_load_si256((__m256i*)&(d[i]));
      auto tmpgt1_ = _mm256_cmpgt_epi32(data_, lim1_); // gt returns 0 or 0xffff..f
      auto count1_ = _mm256_srli_epi32(tmpgt1_, 31); // only keep largest bit
      counter11_ = _mm256_add_epi32(count1_, counter11_); // increment
      auto tmpgt2_ = _mm256_cmpgt_epi32(data_, lim2_);
      auto count2_ = _mm256_srli_epi32(tmpgt2_, 31); // only keep largest bit
      counter21_ = _mm256_add_epi32(count2_, counter21_); // increment
    }

    {
      auto data2_ = _mm256_load_si256((__m256i*)&(d[i+k_vecsize]));
      auto tmpgt1_ = _mm256_cmpgt_epi32(data2_, lim1_); // gt returns 0 or 0xffff..f
      auto count1_ = _mm256_srli_epi32(tmpgt1_, 31); // only keep largest bit
      counter12_ = _mm256_add_epi32(count1_, counter12_); // increment
      auto tmpgt2_ = _mm256_cmpgt_epi32(data2_, lim2_);
      auto count2_ = _mm256_srli_epi32(tmpgt2_, 31); // only keep largest bit
      counter22_ = _mm256_add_epi32(count2_, counter22_); // increment
    }
  }

  auto counter11 = sum_lanes_8(counter11_);
  auto counter21 = sum_lanes_8(counter21_);
  auto counter12 = sum_lanes_8(counter12_);
  auto counter22 = sum_lanes_8(counter22_);

  auto counter1 = counter11 + counter12;
  auto counter2 = counter21 + counter22;
  
  return make_tuple(len - counter1, counter1 - counter2, counter2);
}


/* based on tpch q19
	 the main idea is that the predicate combinations are different. 
*/
int q19lite_all_masked(const lineitem_parts &d, q19params p)
{

	using namespace tbb;
	
	auto body = 	[&](const auto & range, int64_t init) -> int {
	
		int64_t total = init;
		for (int i = range.begin(); i < range.end(); ++i) {
			int64_t mask =
				(d.brand[i] == p.brand1 &
				 d.container[i] == p.container1 &
				 d.quantity[i] < p.max_quantity1 ) |
				(d.brand[i] == p.brand2 &
				 d.container[i] == p.container2 &
				 d.quantity[i] < p.max_quantity2 );

			total += (~(mask-1)) &  (((int64_t)d.eprice[i]) * (100 - d.discount[i]));
		}

		return total;
	};

	return parallel_reduce(blocked_range<size_t>(0, d.len, 1<<10), 0,
									body, [](auto x, auto y) { return x + y; });

}


int q19lite_all_branched (const lineitem_parts &d, q19params p) {
	using namespace tbb;
	
	auto body = 	[&](const auto & range, int64_t init) -> int {

		int64_t total = init;
		for (int i = range.begin(); i < range.end(); ++i) {

			int mask =
			(d.brand[i] == p.brand1 &&
			 d.container[i] == p.container1 &&
			 d.quantity[i] < p.max_quantity1  ) || 
			(d.brand[i] == p.brand2 &&
			 d.container[i] == p.container2  &&
			 d.quantity[i] < p.max_quantity2 );
		
			if (mask)
				{
					total +=  ((int64_t)d.eprice[i]) * (100 - d.discount[i]);
				}

		}

		return  total;
	};

	return parallel_reduce(blocked_range<size_t>(0, d.len, 1<<10), 0,
									body, [](auto x, auto y) { return x + y; });

}

	

