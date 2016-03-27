#include "gtest/gtest.h"
#include <array>
#include <algorithm>
#include "common.h"
#include "impl_helper.h"

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
	{98,98,98,98,98,98,98,98};

__declspec (align(64)) data_t brand_easy[] =
	{1, 1, 1, 1, 1, 1, 1, 1};
__declspec (align(64)) data_t container_easy[] =
	{1, 1, 1, 1, 1, 1, 1, 1};
__declspec (align(64)) data_t quantity_easy[] =
	{1, 1, 1, 1, 1, 1, 1, 1};


// should qualify:
// 1, 1, 0, 0, 1, 0, 0, 0
q19params test_params1 =  {
	.brand = 1,
	.container = 1,
	.max_quantity = 11,
	.min_quantity = 0,
};

q19params test_params2 = {
	.brand = 2,
	.container = 4,
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
		
	auto result = f(d, test_params1);
	ASSERT_EQ(1, result.count);
	ASSERT_EQ(2, result.sum);
}

template <typename F> void testq19_easy(F f){
	lineitem_parts d;
	d.len = 8;
	d.eprice = eprice;
	d.discount = discount;

	d.quantity = quantity_easy;
	d.container = container_easy;
	d.brand = brand_easy;
		
	auto result = f(d, test_params1);
	ASSERT_EQ(8, result.count);
	ASSERT_EQ(16, result.sum);
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

TEST(q19lite, gather) {
	testq19(q19lite_gather);
}

TEST(q19lite, gather_easy) {
	testq19_easy(q19lite_gather);
}

TEST(vector, vec){
	int16_t g1[] = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116};
	Vec16s grp1;
	grp1.load(g1);
	Vec16s grp1orig = grp1;
	
	int16_t g2[] =  {200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216};
	Vec16s grp2;
	grp2.load(g2);
	Vec16s grp2orig = grp2;

	bool done = false;	
	while (!done) {
	 cout << grp1;
	 cout << grp2;

	 Vec16s grp1old = grp1;
	 Vec16s grp2old = grp2;

	 grp1 = _mm256_unpacklo_epi16(grp1old, grp2old);
	 grp2 = _mm256_unpackhi_epi16(grp1old, grp2old);

	 cout << "--------" << endl;
	 done = horizontal_and(grp1 == grp1orig && grp2 == grp2orig);
 }

 cout << grp1;
 cout << grp2;
}

TEST(gather, test){
	SALIGN data_t vec[] = {0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,
												 200,210,220,230,240,250,260,270,280,290,2100,2110,2120,2130,2140,2150};
	SALIGN uint32_t pos[] = {0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30};
	SALIGN data_t expected_raw[] = {0,20,40,60,80,100,120,140,200,220,240,260,280,2100,2120,2140};
	
	auto res = gather(pos, vec);

	vec_t actual(res);
	vec_t expected;
	expected.load(expected_raw);

	cout << expected;
	cout << res;

	for (int i = 0; i < k_elts_per_vec; ++i) {
		ASSERT_EQ(expected[i], actual[i]);
	}	
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
