#include <utility>
#include "common.h"
std::tuple<int,int,int>
  count_naive(int *d, int len, int lim1, int lim2);

std::tuple<int,int,int>
  count_mask(int *d, int len, int lim1, int lim2);

std::tuple<int,int,int>
	count_mask_2unroll(int *d, int len, int lim1, int lim2);


struct lineitem_parts {
	size_t len {};

	int8_t * brand {};
	int8_t * container {};
	int32_t * quantity {};
	
	int32_t * eprice {};
	int8_t * discount {};
};


struct q19row {
	int8_t brand;
	int8_t container;
	int8_t discount;

	int32_t eprice;
	int32_t quantity;
};


void col_to_row(const lineitem_parts & l, q19row *output);
void row_to_col(const q19row *output, lineitem_parts &l);

struct q19params  {
	int8_t brand;
	int8_t container[4] ;
	int32_t max_quantity ;
	int32_t min_quantity ;
};

struct q19res {
	int64_t sum;
	int64_t count;
};

inline lineitem_parts alloc_lineitem_parts (size_t len)
{
	lineitem_parts ans {};

	ans.len = len;
	ans.brand = allocate_aligned<int8_t>(len).release();
	ans.container = allocate_aligned<int8_t>(len).release();
	ans.discount = allocate_aligned<int8_t>(len).release();
	ans.eprice = allocate_aligned<int32_t>(len).release();
	ans.quantity = allocate_aligned<int32_t>(len).release();
	return ans;
}




q19res q19lite_all_masked(const lineitem_parts &, q19params, q19params, q19params);
q19res q19lite_all_branched(const lineitem_parts &, q19params, q19params, q19params);
