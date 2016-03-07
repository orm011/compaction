#include "gtest/gtest.h"
#include "impl.h"
#include <array>
#include <algorithm>

using namespace std;

array<int, 16>
data({0,0,0,0,
      2,2,2,2,
      4,4,4,4,
      8,8,8,8
      });

template <typename F> void testfun(F * fun){
  auto data_ptr  = (int*)(aligned_alloc(32, 16*sizeof(int)));
  copy(data.begin(), data.end(), data_ptr);
  
  auto ans = fun(data_ptr, data.size(), 2, 8);
  ASSERT_EQ(8, get<0>(ans));
  ASSERT_EQ(8, get<1>(ans));
  ASSERT_EQ(0, get<2>(ans));

  ans = fun(data_ptr, data.size(), 2, 4);
  ASSERT_EQ(8, get<0>(ans));
  ASSERT_EQ(4, get<1>(ans));
  ASSERT_EQ(4, get<2>(ans));
}

TEST(count, count_naive){
  testfun(count_naive);
}

TEST(count, count_mask){
  testfun(count_mask);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
