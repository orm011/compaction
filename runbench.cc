#include <gflags/gflags.h>
#include <benchmark/benchmark.h>
#include <tuple>
#include <stdlib.h>
#include "common.h"
#include "impl.h"

DEFINE_int32(array_size_ints, 1<<10, "data size");
DEFINE_int32(limit_lower, 64, "lower limit");
DEFINE_int32(limit_upper, 96, "upper limit");

const int k_max = 128;
const int middle = k_max >> 1;
const int quarter = middle >> 1;
const int threeq = middle + quarter;

using namespace std;

void init_data(int *d, int len)
{
  unsigned int seed = 0;
  for (int i =0 ; i < len; ++i){
    d[i] = rand_r(&seed) % k_max;
  }
}

template <typename Func> void bm_template(benchmark::State & state, Func f){
  auto data_unq = allocate_aligned<int>(FLAGS_array_size_ints);

  auto data = data_unq.get();
  init_data(data, FLAGS_array_size_ints);

  tuple<int, int, int> expected = count_naive(data, FLAGS_array_size_ints, FLAGS_limit_lower, FLAGS_limit_upper);

  tuple<int, int, int> rec;
  while (state.KeepRunning()) {
    rec = f(data, FLAGS_array_size_ints, FLAGS_limit_lower, FLAGS_limit_upper);
  } 

  if (get<0>(expected) != get<0>(rec)){
    cout << "output mismatch at elt 0";
    if (get<1>(expected) != get<1>(rec)){
      cout << "output mismatch at 1";
    } if (get<2>(expected) != get<2>(rec)) {
      cout << "output mistmatch at 2";
    }
  }
  
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


BENCHMARK(bm_count_naive);
BENCHMARK(bm_count_mask);
BENCHMARK(bm_count_mask_2unroll);

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}

//BENCHMARK_MAIN();
