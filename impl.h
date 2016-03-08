#include <utility>

std::tuple<int,int,int>
  count_naive(int *d, int len, int lim1, int lim2);

std::tuple<int,int,int>
  count_mask(int *d, int len, int lim1, int lim2);

std::tuple<int,int,int>
	count_mask_2unroll(int *d, int len, int lim1, int lim2);
