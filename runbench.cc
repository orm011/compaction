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
DEFINE_bool(sorted, false, "sorted");


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

template <typename T, typename C> void init_data(T *d, size_t len, C max)
{

	using namespace tbb;
	
	auto body = [&](const auto &r) {
		unsigned int seed = r.begin(); // make the seed unique by range.
		for (int i = r.begin(); i < r.end(); ++i) {
			d[i] = rand_r(&seed) % max;
		}
	};
	
	parallel_for(blocked_range<size_t>(0, len, 1<<12), body);
}


q19res q19_expected = {-1, -1};

q19params params1 =  {
	.brand = 1,
	.container = {1,2,3,4},
	.max_quantity = 11,
	.min_quantity = 1
};

q19params params2 = {
	.brand = 2,
	.container = {5,6,7,8},
	.max_quantity = 15,
	.min_quantity = 5
};

q19params params3 = {
	.brand = 3,
	.container = {9,10,11,12},
	.max_quantity = 20,
	.min_quantity = 10
};


lineitem_parts g_q19data;

void init_q19data() {
	using namespace tbb;
	vector<int> max_values({10, 15, 20, 10, 90});
	/*brand, container, quantity, eprice, discount */
	g_q19data = alloc_lineitem_parts(FLAGS_array_size_ints);
	
	init_data(g_q19data.brand, FLAGS_array_size_ints, max_values[0]);
	init_data(g_q19data.container, FLAGS_array_size_ints, max_values[1]);
	init_data(g_q19data.quantity, FLAGS_array_size_ints, max_values[2]);
	init_data(g_q19data.eprice, FLAGS_array_size_ints, max_values[2]);
	init_data(g_q19data.discount, FLAGS_array_size_ints, max_values[4]);

	if (FLAGS_sorted) {
		auto rows = allocate_aligned<q19row>(g_q19data.len);
		col_to_row(g_q19data, rows.get());
		auto compareQ19Row = [](const q19row &l, const q19row &r){
			return
			((l.brand < r.brand)) ||
			((l.brand == r.brand) && (l.container < r.container)) ||
			((l.brand == r.brand) && (l.container == r.container) && (l.quantity < r.quantity));		
		};

		parallel_sort(rows.get(), rows.get() + g_q19data.len, compareQ19Row);
		row_to_col(rows.get(), g_q19data);
	}
}

template <typename Func> void q19_template(benchmark::State & state, Func f) {


	if (!g_q19data.brand){
		init_q19data();
		ASSERT_EQ(q19_expected.sum, -1);
	}
	
  if ( q19_expected.count < 0) {
		tbb::task_scheduler_init init_disable(1); // reference always runs serially
		q19_expected = q19lite_all_branched(g_q19data, params1, params2, params3);
		cerr << "selectivity: " << q19_expected.count << "/" << g_q19data.len << " (" << ((double)(q19_expected.count))/g_q19data.len  << ")" << endl;
	}


	q19res res = {0,0};
  while (state.KeepRunning()) {
    res = f(g_q19data, params1, params2, params3);
  }

	
	ASSERT_EQ(q19_expected.count, res.count);
	ASSERT_EQ(q19_expected.sum, res.sum);
}

void bm_q19lite_all_masked(benchmark::State & state) {
  q19_template(state, q19lite_all_masked);
}


void bm_q19lite_all_branched(benchmark::State & state) {
  q19_template(state, q19lite_all_branched);
}

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
