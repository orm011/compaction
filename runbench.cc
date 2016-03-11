#include <gflags/gflags.h>
#include <benchmark/benchmark.h>
#include <tuple>
#include <stdlib.h>
#include "common.h"
#include "impl.h"
#include "gtest/gtest.h"
#include <string>
#include <tbb/tbb.h>

using namespace std;


DEFINE_int32(array_size_ints, 1<<10, "data size (num elements)");
DEFINE_int32(array_size_mb, -1, "data size (MB)");
DEFINE_int32(limit_lower, 64, "lower limit");
DEFINE_int32(limit_upper, 96, "upper limit");
DEFINE_int32(threads, 4, "upper limit");

DEFINE_string(benchmark_filter, ".*", "filter regex");
DEFINE_int32(benchmark_repetitions, 1, "repetitions");
DEFINE_int32(v, 1, "verbosity");
DEFINE_double(benchmark_min_time, 1, "min time seconds");
DEFINE_string(benchmark_format,"tabular", "<tabular|json|csv>)");
DEFINE_bool(benchmark_list_tests, false, "{true|false}");


const int k_max = 128;
const int middle = k_max >> 1;
const int quarter = middle >> 1;
const int threeq = middle + quarter;



void init_data(int *d, int len, int max)
{
  unsigned int seed = 0;
  for (int i =0 ; i < len; ++i){
    d[i] = rand_r(&seed) % max;
  }
}

template <typename Func> void bm_template(benchmark::State & state, Func f){
  auto data_unq = allocate_aligned<int>(FLAGS_array_size_ints);
  auto data = data_unq.get();
  init_data(data, FLAGS_array_size_ints, k_max);

  tuple<int, int, int> expected = count_naive(data, FLAGS_array_size_ints, FLAGS_limit_lower, FLAGS_limit_upper);

  tuple<int, int, int> rec {};
  while (state.KeepRunning()) {
    rec = f(data, FLAGS_array_size_ints, FLAGS_limit_lower, FLAGS_limit_upper);
  } 

	ASSERT_EQ(get<0>(expected), get<0>(rec));
	ASSERT_EQ(get<1>(expected), get<1>(rec));
  ASSERT_EQ(get<2>(expected), get<2>(rec));
}



void bm_count_naive(benchmark::State & state) {
  bm_template(state, count_naive);
}

void bm_count_mask(benchmark::State & state) {
  bm_template(state, count_mask);
}

void bm_count_mask_2unroll(benchmark::State & state) {
  bm_template(state, count_mask_2unroll);
}


int q19_expected = -1;
q19params params =  {
	.brand1 = 1,
	.container1 = 1,
	.max_quantity1 = 11,
	
	.brand2 = 2,
	.container2 = 4,
	.max_quantity2 = 15,
};

lineitem_parts g_q19data;

void init_q19data() {
	vector<int> max_values({4, 6, 20, 10, 90});
	/*brand, container, quantity, eprice, discount */
	vector<aligned_ptr<int>> alloc;
	for (int i =0;i < max_values.size(); ++i) {
		alloc.push_back(allocate_aligned<int>(FLAGS_array_size_ints));
		init_data(alloc.back().get(), FLAGS_array_size_ints, max_values[i]);
	}

	g_q19data.len = FLAGS_array_size_ints;
	g_q19data.brand = alloc[0].release();
	g_q19data.container = alloc[1].release();
	g_q19data.quantity = alloc[2].release();
	g_q19data.eprice = alloc[3].release();
	g_q19data.discount = alloc[4].release();
}

template <typename Func> void q19_template(benchmark::State & state, Func f) {


	if (!g_q19data.brand){
		init_q19data();
		q19_expected = -1;
	}
	
  if ( q19_expected < 0) {
		q19_expected = q19lite_all_branched(g_q19data, params);
	}


	int res = 0;
  while (state.KeepRunning()) {
    res = f(g_q19data, params);
  }

	ASSERT_EQ(q19_expected, res);
}

void bm_q19lite_all_masked(benchmark::State & state) {
  q19_template(state, q19lite_all_masked);
}


void bm_q19lite_all_branched(benchmark::State & state) {
  q19_template(state, q19lite_all_branched);
}


BENCHMARK(bm_count_naive);
BENCHMARK(bm_count_mask);
BENCHMARK(bm_count_mask_2unroll);
BENCHMARK(bm_q19lite_all_masked);
BENCHMARK(bm_q19lite_all_branched);

int main(int argc, char** argv) {
	
	gflags::SetUsageMessage("usage");	
	gflags::ParseCommandLineFlags(&argc, &argv, false);

	if (FLAGS_array_size_mb > 0) {
		FLAGS_array_size_ints = FLAGS_array_size_mb * ((1 << 20) >> 2); // 4 bytes per int
		cout << "NOTE: Array size set to " << FLAGS_array_size_ints << " int elements " << endl;
	}	
	
  ::benchmark::Initialize(&argc, argv);
	tbb::task_scheduler_init init(FLAGS_threads);
  ::benchmark::RunSpecifiedBenchmarks();
}
