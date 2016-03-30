#include "gtest/gtest.h"
#include <array>
#include <algorithm>
#include "common.h"
#include "impl_helper.h"
#include <set>

using namespace std;

__declspec (align(64)) data_t brand[32] =
	{ 1, 2, 3, 4, 2, 1, 2, 1,
	  1, 2, 3, 4, 2, 1, 2, 1,
		1, 2, 3, 4, 2, 1, 2, 1,
		1, 2, 3, 4, 2, 1, 2, 1,};

__declspec (align(64)) data_t container[32] =
	{ 1, 4, 3, 4, 4, 3, 2, 1,
		1, 4, 3, 4, 4, 3, 2, 1,
		1, 4, 3, 4, 4, 3, 2, 1,
		1, 4, 3, 4, 4, 3, 2, 1};

__declspec (align(64)) data_t quantity[32] =
	{ 9,10,11,12,13,14,15,16,
		9,10,11,12,13,14,15,16,
		9,10,11,12,13,14,15,16,
		9,10,11,12,13,14,15,16};

__declspec (align(64)) data_t eprice[32] =
	{ 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1,};

__declspec (align(64)) data_t discount[32] =
	{98,98,98,98,98,98,98,98,
	 98,98,98,98,98,98,98,98,
	 98,98,98,98,98,98,98,98,
	 98,98,98,98,98,98,98,98,};

__declspec (align(64)) data_t brand_easy[32] =
	{ 1, 1, 1, 1, 1, 1, 1, 1,
	  1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1,};
		
__declspec (align(64)) data_t container_easy[32] =
	{ 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1,};

__declspec (align(64)) data_t quantity_easy[32] =
	{ 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1,};


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
	d.len = 32;
	d.eprice = eprice;
	d.discount = discount;
	d.quantity = quantity;

	d.container = container;
	d.brand = brand;
		
	auto result = f(d, test_params1);
	ASSERT_EQ(4, result.count);
	ASSERT_EQ(8, result.sum);
}

template <typename F> void testq19_easy(F f){
	lineitem_parts d;
	d.len = 32;
	d.eprice = eprice;
	d.discount = discount;

	d.quantity = quantity_easy;
	d.container = container_easy;
	d.brand = brand_easy;
		
	auto result = f(d, test_params1);
	ASSERT_EQ(32, result.count);
	ASSERT_EQ(64, result.sum);
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
		//	 cout << grp1;
		//cout << grp2;

	 Vec16s grp1old = grp1;
	 Vec16s grp2old = grp2;

	 //grp1 = _mm256_unpacklo_epi16(grp1old, grp2old);
	 //grp2 = _mm256_unpackhi_epi16(grp1old, grp2old);
	 //cout << "--------" << endl;
	 done = horizontal_and(grp1 == grp1orig && grp2 == grp2orig);
 }

	//cout << grp1;
	//cout << grp2;
}

TEST(util, weak_gather){
	
	SALIGN data_t vec[64];
	for (int i =0; i < 64; ++i){
		vec[i] = (int8_t)i;
	}

	SALIGN uint32_t pos[32];
	for (int i = 0; i < 32; ++i){
		pos[i] = i << 1;
	}

	auto res = gather(pos, vec);

	std::set<data_t> expected_set;
	for (int i = 0; i < k_elts_per_vec; ++i){
		expected_set.emplace(vec[pos[i]]);
	}
	ASSERT_EQ(k_elts_per_vec, expected_set.size());


	vec_t actual;
	actual.load(&res);
	std::set<data_t> actual_set;
	for (int i = 0; i < k_elts_per_vec; ++i) {
		actual_set.emplace(actual[i]);
	}
	ASSERT_EQ(k_elts_per_vec, actual_set.size()); // also no repeats here
	
	//cout << res;
	
	for (auto &elt : actual_set) {
		ASSERT_TRUE(expected_set.count(elt));
	}
}


TEST(util, basicVec4qVec){
	Vec4q vr(0,1,2,3);
	
	Vec4qVec<1> v;
	v.arr[0].load(&vr);

	Vec4qVec<2> w(0);
	w.assign_at<0>(v);
	for (int i =0; i < 4; ++i){
		ASSERT_EQ(v.arr[0][i], w.arr[0][i]);
		ASSERT_EQ(v.arr[0][i], w[i]);
	}

	for (int i = 0; i < 4;  ++i){
		ASSERT_EQ(0, w[i + 4]);
	}

	Vec4q v2r(1,3,5,7);
	Vec4qVec<1> v2;
	v2.arr[0].load(&v2r);
	w.assign_at<1>(v2);
	
	for (int i =0; i < 4; ++i) {
		ASSERT_EQ(v.arr[0][i], w[i]);
		ASSERT_EQ(v2.arr[0][i], w[i + 4]);
	}
}


TEST(util, extend_int){
	int32_t  ints_raw[64];
	for (int i = 0; i < 64; ++i) {
		ints_raw[i] = i;
	}

	Vec8i ints;
	ints.load(ints_raw);

	Vec4qVec<2> expanded = extend(ints);

	for (int i = 0; i < 8; ++i){
		ASSERT_EQ(ints_raw[i], expanded[i]);
	}
}



TEST(util, extend_char){
	int8_t  chars_raw[64];
	for (int i = 0; i < 64; ++i) {
		chars_raw[i] = i;
	}

	Vec32c chars;
	chars.load(chars_raw);

	Vec4qVec<8> expanded = extend(chars);

	for (int i = 0; i < 32; ++i){
		ASSERT_EQ(chars_raw[i], expanded.arr[i/4][i%4]);
	}
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
