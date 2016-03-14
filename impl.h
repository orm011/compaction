#include <utility>

std::tuple<int,int,int>
  count_naive(int *d, int len, int lim1, int lim2);

std::tuple<int,int,int>
  count_mask(int *d, int len, int lim1, int lim2);

std::tuple<int,int,int>
	count_mask_2unroll(int *d, int len, int lim1, int lim2);


struct lineitem_parts {
	size_t len {};

	int * brand {};
	int * container {};
	int * quantity {};
	
	int * eprice {};
	int * discount {};
};

struct q19params  {
	int brand;
	int container[4] ;
	int max_quantity ;
	int min_quantity ;
};

struct q19res {
	int64_t sum;

	int64_t count;
};


q19res q19lite_all_masked(const lineitem_parts &, q19params, q19params, q19params);
q19res q19lite_all_branched(const lineitem_parts &, q19params, q19params, q19params);
