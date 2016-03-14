#include "gtest/gtest.h"
#include "impl.h"
#include <array>
#include <algorithm>
#include "common.h"

using namespace std;

array<int, 16>
data({0,0,0,0,
      2,2,2,2,
      4,4,4,4,
      8,8,8,8
      });

template <typename F> void testfun(F * fun) {
  auto data_ptr_unq  = allocate_aligned<int>(32);
  auto data_ptr = data_ptr_unq.get();
  
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




TEST(count, naive){
  testfun(count_naive);
}

TEST(count, mask){
  testfun(count_mask);
}

TEST(count, mask_2unroll){
	testfun(count_mask_2unroll);
}

__declspec (align(64)) int brand[] =
{ 1, 2, 3, 4, 2, 3, 2, 1};

__declspec (align(64)) int container[] =
{ 1, 4, 3, 4, 4, 3, 2, 1};

__declspec (align(64)) int quantity[] =
{ 9,10,11,12,13,14,15,16};

__declspec (align(64)) int eprice[] =
{ 1, 1, 1, 1, 1, 1, 1, 1};

__declspec (align(64)) int discount[] =
{98,98,98,98,98,98,98};

// should qualify:
// 1, 1, 0, 0, 1, 0, 0, 0
q19params test_params1 =  {
	.brand = 1,
	.container = {1,1,1,1},
	.max_quantity = 11,
	.min_quantity = 0,
};

q19params test_params2 = {
	.brand = 2,
	.container = {4,4,4,4},
	.max_quantity = 15,
	.min_quantity = 0,
};

template <typename F> void testq19(F f){
	lineitem_parts d;
	d.len = 8;
	d.eprice = eprice;
	d.discount = discount;
	d.quantity = quantity;

	d.container = container;
	d.brand = brand;
		
	auto result = f(d, test_params1, test_params2, test_params2);
	ASSERT_EQ(3, result.count);
	ASSERT_EQ(6, result.sum);
}


TEST(q19lite, all_masked) {
	testq19(q19lite_all_masked);
}

TEST(q19lite, all_branched) {
	testq19(q19lite_all_branched);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
