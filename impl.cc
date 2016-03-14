#include "common.h"
#include "impl.h"
#include <tbb/tbb.h>
#include <gflags/gflags.h>

DEFINE_int32(grain_size, 1 << 12, "minimum amount of work (num array elts)");

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

#define Q19PRED($d,$i,$p,$AND,$OR) \
	((($d).brand[($i)] == ($p).brand) $AND \
	 (($d).quantity[($i)] < ($p).max_quantity) $AND \
	 (($d).quantity[($i)] >= ($p).min_quantity) $AND \
	 ( ($d).container[($i)] == ($p).container[0] $OR \
		 ($d).container[($i)] == ($p).container[1] $OR \
		 ($d).container[($i)] == ($p).container[2] $OR \
		 ($d).container[($i)] == ($p).container[3] ))


static const q19res  init = {.count = 0, .sum = 0};
static const auto addq19 = [](const q19res &x, const q19res & y) -> q19res { q19res ans; ans.count = x.count + y.count;  ans.sum = (x.sum + y.sum); return ans; };

/* based on tpch q19
	 the main idea is that the predicate combinations are different. 
*/
q19res q19lite_all_masked(const lineitem_parts &d, q19params p1, q19params p2, q19params p3)
{

	using namespace tbb;
	
	auto body = 	[&](const auto & range, const auto & init)  {
	
		auto total = init;
		for (int i = range.begin(); i < range.end(); ++i) {
			int64_t mask = Q19PRED(d, i, p1, &, |) | Q19PRED(d,i,p2,&,|) | Q19PRED(d,i,p3,&,|);
			total.sum += (~(mask-1)) &  (((int64_t)d.eprice[i]) * (100 - d.discount[i]));
			total.count += mask;
		}

		return total;
	};


	return parallel_reduce(blocked_range<size_t>(0, d.len, FLAGS_grain_size), init, body, addq19);
}


q19res q19lite_all_branched (const lineitem_parts &d, q19params p1, q19params p2, q19params p3) {
	using namespace tbb;
	
	auto body = 	[&](const auto & range, const auto & init)  {
		q19res total = init;
		for (int i = range.begin(); i < range.end(); ++i) {
			int64_t mask = Q19PRED(d, i, p1, &&, ||) || Q19PRED(d,i,p2,&&,||) || Q19PRED(d,i,p3,&&,||);

			if (mask)
				{
					total.sum +=  ((int64_t)d.eprice[i]) * (100 - d.discount[i]);
					total.count += 1;
				}
		}

		return  total;
	};

	return parallel_reduce(blocked_range<size_t>(0, d.len, FLAGS_grain_size), init, body, addq19);

}
