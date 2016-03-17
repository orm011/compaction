#include "common.h"
#include "impl.h"
#include <tbb/tbb.h>
#include <gflags/gflags.h>

using namespace std;
DEFINE_int32(grain_size, 1 << 12, "minimum amount of work (num array elts)");

void col_to_row(const lineitem_parts & columns, q19row *rows){
	using namespace tbb;
	auto body = [&](const auto & r){
		for (int i = r.begin(); i < r.end(); ++i){
			rows[i].brand = columns.brand[i];
			rows[i].container = columns.container[i];
			rows[i].discount = columns.discount[i];
			rows[i].eprice = columns.eprice[i];
			rows[i].quantity = columns.quantity[i];
		}
	};

	parallel_for(blocked_range<size_t>(0, columns.len, FLAGS_grain_size),  body);
}


void row_to_col(const q19row *rows, lineitem_parts &columns){
	using namespace tbb;
	auto body = [&](const auto & r){
		for (int i = r.begin(); i < r.end(); ++i){
			columns.brand[i] = rows[i].brand;
			columns.container[i] = rows[i].container;
			columns.discount[i] = rows[i].discount;
			columns.eprice[i] = rows[i].eprice;
			columns.quantity[i] = rows[i].quantity;
		}
	};

	parallel_for(blocked_range<size_t>(0, columns.len, FLAGS_grain_size), body);
}




#define Q19PRED_INNER($d,$i,$p,$AND1, $AND2, $AND3, $OR)	\
	((($d).brand[($i)] == ($p).brand) $AND1 \
	 (($d).quantity[($i)] < ($p).max_quantity) $AND2 \
	 (($d).quantity[($i)] >= ($p).min_quantity) $AND3 \
	 ( ($d).container[($i)] == ($p).container[0] $OR \
		 ($d).container[($i)] == ($p).container[1] $OR \
		 ($d).container[($i)] == ($p).container[2] $OR \
		 ($d).container[($i)] == ($p).container[3] ))

#define Q19PRED($d,$i,$p,$AND, $OR)				\
	Q19PRED_INNER($d,$i,$p,$AND, $AND, $AND, $OR)

static const q19res  init = {.count = 0, .sum = 0};
static const auto addq19 = [](const q19res &x, const q19res & y) -> q19res { q19res ans; ans.count = x.count + y.count;  ans.sum = (x.sum + y.sum); return ans; };

/* based on tpch q19
	 the main idea is that the predicate combinations are different. 
*/
q19res q19lite_all_masked_vectorized(const lineitem_parts &d, q19params p1, q19params p2, q19params p3)
{

	using namespace tbb;
	
	auto body = 	[&](const auto & range, const auto & init)  {
	
		auto total = init;
		#pragma  vector always
		for (int i = range.begin(); i < range.end(); ++i) {
			int64_t mask = Q19PRED(d, i, p1, &, |) | Q19PRED(d,i,p2,&,|) | Q19PRED(d,i,p3,&,|);
			total.sum += (~(mask-1)) &  (((int64_t)d.eprice[i]) * (100 - d.discount[i]));
			total.count += mask;
		}

		return total;
	};


	return parallel_reduce(blocked_range<size_t>(0, d.len, FLAGS_grain_size), init, body, addq19);
}


q19res q19lite_all_masked_scalar(const lineitem_parts &d, q19params p1, q19params p2, q19params p3)
{

	using namespace tbb;
	
	auto body = 	[&](const auto & range, const auto & init)  {
	
		auto total = init;
		#pragma novector
		for (int i = range.begin(); i < range.end(); ++i) {
			int64_t mask = Q19PRED(d, i, p1, &, |) | Q19PRED(d,i,p2,&,|) | Q19PRED(d,i,p3,&,|);
			total.sum += (~(mask-1)) &  (((int64_t)d.eprice[i]) * (100 - d.discount[i]));
			total.count += mask;
		}

		return total;
	};


	return parallel_reduce(blocked_range<size_t>(0, d.len, FLAGS_grain_size), init, body, addq19);
}



q19res q19lite_all_branched (const lineitem_parts &d, q19params p1, q19params p2, q19params p3) {
	using namespace tbb;
	
	auto body = 	[&](const auto & range, const auto & init)  {
		q19res total = init;
		for (int i = range.begin(); i < range.end(); ++i) {
			int64_t mask = Q19PRED(d, i, p1, &&, ||) || Q19PRED(d,i,p2,&&,||) || Q19PRED(d,i,p3,&&,||);

			if (mask)
				{
					total.sum +=  ((int64_t)d.eprice[i]) * (100 - d.discount[i]);
					total.count += 1;
				}
		}

		return  total;
	};

	return parallel_reduce(blocked_range<size_t>(0, d.len, FLAGS_grain_size), init, body, addq19);

}

q19res q19lite_vectorized_assume_sorted(const lineitem_parts &d, q19params p1, q19params p2, q19params p3)
{

	(void)p2; (void)p3;
	using namespace tbb;
	auto body = 	[&](const auto & range, const auto & init)  {
		auto total = init;
		#pragma  vector always
		for (int i = range.begin(); i < range.end(); ++i) {
			
			int64_t mask = Q19PRED_INNER(d, i, p1, &&, &, &, |);

			if (mask) {
				total.sum += ((int64_t)d.eprice[i]) * (100 - d.discount[i]);
				total.count += mask;
			}
		}

		return total;
	};

	return parallel_reduce(blocked_range<size_t>(0, d.len, FLAGS_grain_size), init, body, addq19);
}
