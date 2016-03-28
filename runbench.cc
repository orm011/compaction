#include <gflags/gflags.h>
#include <benchmark/benchmark.h>
#include <tuple>
#include <stdlib.h>
#include "common.h"
#include "impl_helper.h"
#include "gtest/gtest.h"
#include <string>
#include <tbb/tbb.h>
#include <cpucounters.h>
#include <signal.h>
#include <types.h>
#include <stdlib.h>
#include <limits>		
#include <random>
#include <functional>
#include <cstdio>

using namespace std;

DEFINE_int32(array_size_elts, 1<<10, "data size (num elements)");
DEFINE_int32(array_size_mb, -1, "data size (MB)");
DEFINE_int32(limit_lower, 64, "lower limit");
DEFINE_int32(limit_upper, 96, "upper limit");
DEFINE_int32(num_brands, 100, "selectivity of first predicate is 1/num_brands");

DEFINE_int32(threads, 4, "upper limit");
DEFINE_bool(sorted, false, "sorted");
DEFINE_bool(clustered, false, "{true|false}");

DEFINE_string(benchmark_filter, ".*", "filter regex");
DEFINE_int32(benchmark_repetitions, 1, "repetitions");
DEFINE_int32(v, 1, "verbosity");
DEFINE_double(benchmark_min_time, 1, "min time seconds");
DEFINE_string(benchmark_format,"tabular", "<tabular|json|csv>)");
DEFINE_bool(benchmark_list_tests, false, "{true|false}");
DEFINE_bool(use_pmu_counter, false, "{true|false}");
DEFINE_bool(verify_uniform, false, "{true|false}");


const int k_max = 128;
const int middle = k_max >> 1;
const int quarter = middle >> 1;
const int threeq = middle + quarter;
PCM * m = nullptr;

template <typename T> void init_data(T *d, size_t len, int32_t max)
{

	using namespace tbb;
	if (max > std::numeric_limits<T>::max()){
		throw std::runtime_error("max is too large");
	}

	if (max < 0) {
		throw std::runtime_error("max should be non-negative");
	}

	auto body = [&](const auto &r) {
		// random_device rd;
		// minstd_rand  gen(rd());
		// gen.seed(r.begin());
		// uniform_int_distribution<> dis(0, max);
		uint32_t seed = r.begin();
		
		for (int i = r.begin(); i < r.end(); ++i) {
			d[i] = (T)(rand_r(&seed) % (max+1));
		}
	};
	
	parallel_for(blocked_range<size_t>(0, len, 1<<12), body);

	if (FLAGS_verify_uniform){
		auto check_body = [&](const auto &r, auto init) -> size_t {
			for (int i = r.begin(); i < r.end(); ++i) {
				init += (d[i] == 0);
			}

			return init;
		};
		
		size_t total = parallel_reduce(blocked_range<size_t>(0, len, 1<<12), 0, check_body, [](const auto &a, const auto &b) { return a + b; });

		cout << "count / len = " <<  total << " / " << len << " = " << ((double)total/len) * 100 << " % vs. " << ((double)1/(max+1)) << endl;
	}
}

struct red_res {
	size_t words = 0;
	size_t elts = 0;
};

red_res	operator+(const red_res &a, const red_res &b) {
	red_res ans;
	ans.words = a.words + b.words;
	ans.elts = a.elts + b.elts;
	return ans;
}

template <typename T, typename Pred>
red_res count_words_elts(T *d, size_t len, Pred pred, size_t elts_per_cache_line)
{
	using namespace tbb;
	assert(0 == ((size_t)d) % k_cache_line_size);
	assert(0 == (len % k_elts_per_line));
	
	auto body = [&](const auto &r, const auto &init)  {
		red_res res{};
		
		for ( size_t i = r.begin(); i < r.end(); i+=elts_per_cache_line ) {
			size_t found = 0;
			for (size_t j = 0; j < elts_per_cache_line; ++j) {
				found += pred(d[i + j]);
			}

			res = res + red_res {(found > 0), found};
		}

		return res + init;
	};
	
	return parallel_reduce(blocked_range<size_t>(0, len, 1<<12),
												 red_res(), body, std::plus<red_res>());
}


q19res q19_expected = {-1, -1};


// q19params params2 = {
// 	.brand = 2,
// 	.container = {5,6,7,8},
// 	.max_quantity = 15,
// 	.min_quantity = 5
// };

// q19params params3 = {
// 	.brand = 3,
// 	.container = {9,10,11,12},
// 	.max_quantity = 20,
// 	.min_quantity = 10
// };


lineitem_parts g_q19data;
lineitem_parts g_clustered_q19data;

q19params params1;


void init_q19data() {
	using namespace tbb;
	if (FLAGS_num_brands <= 0) {
		throw std::runtime_error("need at least one brand");
	}
	
	vector<int> num_distinct_values({FLAGS_num_brands, 1, 8, 1024, 101});
	vector<int> max_values;
	for (auto & i: num_distinct_values) {
		max_values.push_back(i - 1);
	}
	
	/*brand, container, quantity, eprice, discount */
	g_q19data = alloc_lineitem_parts(FLAGS_array_size_elts);
	
	init_data(g_q19data.brand, FLAGS_array_size_elts, max_values[0]);
	init_data(g_q19data.container, FLAGS_array_size_elts, max_values[1]);
	init_data(g_q19data.quantity, FLAGS_array_size_elts, max_values[2]);
	
	init_data(g_q19data.eprice, FLAGS_array_size_elts, max_values[3]);
	init_data(g_q19data.discount, FLAGS_array_size_elts, max_values[4]);

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


	params1.brand = 0;
	params1.container = 0;
	params1.max_quantity = 8;
	params1.min_quantity = 0;

	if (FLAGS_clustered){
		g_clustered_q19data = alloc_lineitem_parts(FLAGS_array_size_elts);
		q19lite_cluster(g_q19data, params1, g_clustered_q19data);
	}

}

void report_selectivities(red_res r, size_t data_len){
	auto total_words = data_len/ k_elts_per_line;

	auto word_sel = (((double) r.words) / total_words ) * 100;
	auto word_MB = (r.words * 64) >> 20;
	fprintf(stderr, "cache line level selectivity:\t%llu / %llu = %.1f %% ( %llu MB )\n", r.words, total_words, word_sel, word_MB);

	auto elt_sel = (((double) r.elts) / data_len ) * 100;
	auto elt_MB = (r.elts *sizeof(data_t)) >> 20;
	fprintf(stderr, "row level selectivity:\t%llu / %llu = %.1f %% ( %llu MB )\n", r.elts, data_len, elt_sel, elt_MB);
}

template <typename Func> void q19_template(benchmark::State & state, Func f) {


	if (!g_q19data.brand){
		init_q19data();
		ASSERT_EQ(q19_expected.sum, -1);
	}


  if ( q19_expected.count < 0) {
		tbb::task_scheduler_init init_disable(1); // reference always runs serially
		q19_expected = q19lite_all_branched(g_q19data, params1);


		auto brand_pred = [&](auto elt) { return elt == params1.brand;  };
		auto counts = count_words_elts(g_q19data.brand, g_q19data.len, brand_pred, k_elts_per_line);
		report_selectivities(counts, g_q19data.len);		
		ASSERT_EQ(counts.elts, q19_expected.count); // make sure the other predicates aren't filtering out things for now.

		if (FLAGS_clustered){
			auto counts = count_words_elts(g_clustered_q19data.brand, g_clustered_q19data.len, brand_pred, k_elts_per_line);
			cerr << "stats for clustered data" << endl;
			report_selectivities(counts, g_clustered_q19data.len);		
		}
	}


	q19res res = {0,0};
	SocketCounterState before, after;

	if (FLAGS_use_pmu_counter) {
		before = m->getSocketCounterState(0);
	}

	if (FLAGS_clustered){
		while (state.KeepRunning()) {
			res = f(g_clustered_q19data, params1);
		}
	} else {
		while (state.KeepRunning()) {
			res = f(g_q19data, params1);
		}
	}
	
	if (FLAGS_use_pmu_counter) {
			after = m->getSocketCounterState(0);
			uint64 read =  getBytesReadFromMC (before, after);
			uint64 write =  getBytesWrittenToMC (before, after);
			cout  << "MBs read from MC: " << (read >> 20) << endl;
			cout  << "MBs written to MC: " << (write >> 20) << endl;
	}
	
	
	ASSERT_EQ(q19_expected.count, res.count);
	ASSERT_EQ(q19_expected.sum, res.sum);
}

void bm_q19lite_all_masked_vectorized(benchmark::State & state) {
  q19_template(state, q19lite_all_masked_vectorized);
}

void bm_q19lite_all_masked_scalar(benchmark::State & state) {
  q19_template(state, q19lite_all_masked_scalar);
}

void bm_q19lite_all_branched(benchmark::State & state) {
  q19_template(state, q19lite_all_branched);
}


void bm_q19lite_gather(benchmark::State & state) {
	q19_template(state, q19lite_gather);
}

BENCHMARK(bm_q19lite_all_masked_vectorized);
BENCHMARK(bm_q19lite_all_masked_scalar);
BENCHMARK(bm_q19lite_all_branched);
BENCHMARK(bm_q19lite_gather);

void intHandler(int ) {
	cout << "cleaning up  .... " << endl;
	if (m){
		m->cleanup();
	}
	exit(1);
}

int main(int argc, char** argv) {
	signal(SIGINT, intHandler);


	gflags::SetUsageMessage("usage");	
	gflags::ParseCommandLineFlags(&argc, &argv, false);

	if (FLAGS_array_size_mb > 0) {
		FLAGS_array_size_elts = FLAGS_array_size_mb * ((1 << 20)/sizeof(data_t)); 
		cout << "NOTE: Elt size is " << sizeof(data_t) << ". Array size set to " << FLAGS_array_size_elts << " elements." << endl;
	}	
	
  ::benchmark::Initialize(&argc, argv);
	tbb::task_scheduler_init init(FLAGS_threads);


	if (FLAGS_use_pmu_counter) {
	m = PCM::getInstance();	
	PCM::ErrorCode status = m->program();
	// program counters, and on a failure just exit
	switch (status)
		{
		case PCM::Success:
			break;
		case PCM::MSRAccessDenied:
			cerr << "Access to Intel(r) Performance Counter Monitor has denied (no MSR or PCI CFG space access)." << endl;
			exit(EXIT_FAILURE);
		case PCM::PMUBusy:
			cerr << "Access to Intel(r) Performance Counter Monitor has denied (Performance Monitoring Unit is occupied by other application). Try to stop the application that uses PMU." << endl;
			cerr << "Alternatively you can try to reset PMU configuration at your own risk. Try to reset? (y/n)" << endl;
			char yn;
			std::cin >> yn;
			if ('y' == yn)
				{
					m->resetPMU();
					cerr << "PMU configuration has been reset. Try to rerun the program again." << endl;
				}
			exit(EXIT_FAILURE);
		default:
			cerr << "Access to Intel(r) Performance Counter Monitor has denied (Unknown error)." << endl;
			exit(EXIT_FAILURE);
		}
	}
	
  ::benchmark::RunSpecifiedBenchmarks();

	if (FLAGS_use_pmu_counter){
		m->cleanup();
	}
}
