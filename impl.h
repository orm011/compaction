#include <utility>
#include "common.h"

#ifndef DATA_T
typedef int32_t data_t;
#else
typedef DATA_T data_t;
#endif

static_assert(sizeof(data_t) > 1, "uint8_t cannot have selectivity lower than 1/255");

struct lineitem_parts {
	size_t len {};

	data_t * brand {};
	data_t * container {};
	data_t * quantity {};
	
	data_t * eprice {};
	data_t * discount {};
};


struct q19row {
	data_t brand;
	data_t container;
	data_t discount;

	data_t eprice;
	data_t quantity;
};


void col_to_row(const lineitem_parts & l, q19row *output);
void row_to_col(const q19row *output, lineitem_parts &l);

struct q19params  {
	data_t brand;
	data_t container ;
	data_t max_quantity ;
	data_t min_quantity ;
};

struct q19res {
	int64_t sum;
	int64_t count;
};

inline lineitem_parts alloc_lineitem_parts (size_t len)
{
	lineitem_parts ans {};

	ans.len = len;
	ans.brand = allocate_aligned<data_t>(len).release();
	ans.container = allocate_aligned<data_t>(len).release();
	ans.discount = allocate_aligned<data_t>(len).release();
	ans.eprice = allocate_aligned<data_t>(len).release();
	ans.quantity = allocate_aligned<data_t>(len).release();
	return ans;
}

q19res q19lite_all_masked_scalar(const lineitem_parts &, q19params);
q19res q19lite_all_masked_vectorized(const lineitem_parts &, q19params);
q19res q19lite_all_branched(const lineitem_parts &, q19params);
