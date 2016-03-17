#include "gtest/gtest.h"
#include "impl.h"
#include <array>
#include <algorithm>
#include "common.h"

using namespace std;

__declspec (align(64)) data_t brand[] =
{ 1, 2, 3, 4, 2, 3, 2, 1};

__declspec (align(64)) data_t container[] =
{ 1, 4, 3, 4, 4, 3, 2, 1};

__declspec (align(64)) data_t quantity[] =
{ 9,10,11,12,13,14,15,16};

__declspec (align(64)) data_t eprice[] =
{ 1, 1, 1, 1, 1, 1, 1, 1};

__declspec (align(64)) data_t discount[] =
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

TEST(util, col_to_row_to_col){
	lineitem_parts d;
	d.len = 8;
	d.eprice = eprice;
	d.discount = discount;
	d.quantity = quantity;

	d.container = container;
	d.brand = brand;

	auto rows = allocate_aligned<q19row>(d.len);
	col_to_row(d, rows.get());
	lineitem_parts newcols = alloc_lineitem_parts(d.len);
	row_to_col(rows.get(), newcols);

	ASSERT_EQ(d.len, newcols.len);
	ASSERT_TRUE(std::equal(d.brand, d.brand + d.len, newcols.brand));
	ASSERT_TRUE(std::equal(d.container, d.container + d.len, newcols.container));
	ASSERT_TRUE(std::equal(d.quantity, d.quantity + d.len, newcols.quantity));
	ASSERT_TRUE(std::equal(d.eprice, d.eprice + d.len, newcols.eprice));
	ASSERT_TRUE(std::equal(d.discount, d.discount + d.len, newcols.discount));
}

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


TEST(q19lite, all_masked_scalar) {
	testq19(q19lite_all_masked_scalar);
}

TEST(q19lite, all_masked_vectorized) {
	testq19(q19lite_all_masked_vectorized);
}

TEST(q19lite, all_branched) {
	testq19(q19lite_all_branched);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
