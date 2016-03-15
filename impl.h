#include <utility>

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


q19res q19lite_all_masked(const lineitem_parts &, q19params, q19params, q19params);
q19res q19lite_all_branched(const lineitem_parts &, q19params, q19params, q19params);
