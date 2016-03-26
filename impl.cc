#include "common.h"
#include "impl.h"
#include <tbb/tbb.h>
#include <gflags/gflags.h>

using namespace std;
DEFINE_int32(grain_size, (1 << 14)/sizeof(data_t), "minimum amount of work (num array elts)");

typedef __m256i vec_t;

const static size_t k_buf_size = 2048;
const static size_t k_vec_size = sizeof(vec_t);
static_assert(k_buf_size % k_vec_size == 0, "vector must divide buffer");
static_assert(k_vec_size % sizeof(data_t) == 0, "data_t must divide vector");

const static size_t k_elts_per_vec = k_vec_size/sizeof(data_t);
const static size_t k_elts_per_buf = k_buf_size/sizeof(data_t);

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


#define Q19PRED_INNER($d,$i,$p, $AND1, $AND2, $AND3)	\
	((($d).brand[($i)] == ($p).brand) $AND1						\
	 (($d).quantity[($i)] < ($p).max_quantity) $AND2	\
	 (($d).quantity[($i)] >= ($p).min_quantity) $AND3 \
	 ($d).container[($i)] == ($p).container )

#define Q19PRED($d,$i,$p,$AND)									\
	Q19PRED_INNER($d,$i,$p,$AND, $AND, $AND)

static const q19res  init = {.count = 0, .sum = 0};
static const auto addq19 = [](const q19res &x, const q19res & y) -> q19res { q19res ans; ans.count = x.count + y.count;  ans.sum = (x.sum + y.sum); return ans; };

/* based on tpch q19
	 the main idea is that the predicate combinations are different. 
*/
q19res q19lite_all_masked_vectorized(const lineitem_parts &d, q19params p1)
{

	using namespace tbb;
	
	auto body = 	[&](const auto & range, const auto & init)  {
	
		auto total = init;
		#pragma  vector always
		for (int i = range.begin(); i < range.end(); ++i) {
			int64_t mask = Q19PRED(d, i, p1, &);
			total.sum += (~(mask-1)) &  (((int64_t)d.eprice[i]) * (100 - d.discount[i]));
			total.count += mask;
		}

		return total;
	};

	return parallel_reduce(blocked_range<size_t>(0, d.len, FLAGS_grain_size), init, body, addq19);
}


q19res q19lite_all_masked_scalar(const lineitem_parts &d, q19params p1)
{

	using namespace tbb;
	
	auto body = 	[&](const auto & range, const auto & init)  {
		auto total = init;
		#pragma novector
		for (int i = range.begin(); i < range.end(); ++i) {
			int64_t mask = Q19PRED(d, i, p1, &);
			total.sum += (~(mask-1)) &  (((int64_t)d.eprice[i]) * (100 - d.discount[i]));
			total.count += mask;
		}

		return total;
	};


	return parallel_reduce(blocked_range<size_t>(0, d.len, FLAGS_grain_size), init, body, addq19);
}



q19res q19lite_all_branched (const lineitem_parts &d, q19params p1) {
	using namespace tbb;
	
	auto body = 	[&](const auto & range, const auto & init)  {
		q19res total = init;
		#pragma novector
		for (int i = range.begin(); i < range.end(); ++i) {
			int64_t mask = Q19PRED(d, i, p1, &&);

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



q19res q19lite_gather (const lineitem_parts &d, q19params p1) {
	using namespace tbb;
	
	const auto container_expected = _mm256_set1_epi32(p1.container);
	const auto qty_low = _mm256_set1_epi32(p1.min_quantity - 1); //  bc GThan
	const auto qty_max = _mm256_set1_epi32(p1.max_quantity);
	
	auto body = 	[&](const auto & range, const auto & init)  {
		auto startbrand = &d.brand[range.begin()];
		auto startcontainer = &d.container[range.begin()];
		auto startquantity  = &d.quantity[range.begin()];
		auto starteprice  = &d.eprice[range.begin()];
		auto startdiscount  = &d.discount[range.begin()];
		
		auto acc_total = _mm256_set1_epi32(0);
		auto acc_counts = _mm256_set1_epi32(0);
		const auto hundred_ = _mm256_set1_epi32(100);
		q19res total = init;
		
		__declspec(align(64)) uint32_t buf[k_buf_size] {};
		
		//auto process_buffer = [&](auto buf_size){
			// for (int idx = 0; idx < buf_size; idx += k_elts_per_vec ) {
			// 			auto vindex = _mm256_load_si256((__m256i *)&buf[idx]);
						
			// 			auto containerv =
			// 				_mm256_i32gather_epi32(startcontainer, vindex, sizeof(data_t));
			// 			auto quantityv =
			// 				_mm256_i32gather_epi32(startquantity, vindex, sizeof(data_t));
			// 			auto epricev =
			// 				_mm256_i32gather_epi32(starteprice, vindex, sizeof(data_t));
			// 			auto discountv =
			// 				_mm256_i32gather_epi32(startdiscount, vindex, sizeof(data_t));

			// 			auto containqual = _mm256_cmpeq_epi32 (containerv, container_expected);
			// 			auto geqlow = _mm256_cmpgt_epi32 (quantityv, qty_low);
			// 			auto lthigh = _mm256_cmpgt_epi32 (qty_max, quantityv);
			// 			auto quals1 = _mm256_and_si256 (containqual, geqlow);
			// 			auto mask = _mm256_and_si256 (quals1, lthigh);

			// 			auto counts = _mm256_srli_epi32 (mask, 31);
			// 			acc_counts = _mm256_add_epi32(counts, acc_counts);

			// 			auto minus = _mm256_sub_epi32(hundred_, discountv);
			// 			auto prod = _mm256_mullo_epi32 (epricev, minus);
			// 			auto prod_and = _mm256_and_si256(prod, mask);
			// 			acc_total = _mm256_add_epi32(acc_total, prod_and);						
			// 		}
		//};

		int j = 0;
		for (uint32_t i = 0; i < (range.end() - range.begin()); ++i) {
			buf[j] = i;
			j += (startbrand[i] == p1.brand);

			if (j == k_elts_per_buf) {
				//process_buffer(k_elts_per_buf);
				j = 0;
			}
		}
		
		const auto vec_tail_end = (j / k_elts_per_vec) * k_elts_per_vec ;
		//process_buffer(vec_tail_end);

		for (int idx = vec_tail_end; idx < j; ++idx) {
			auto i = buf[idx];
			auto mask =
			(startquantity[i] >= p1.min_quantity  &&
			 startquantity[i] < p1.max_quantity  &&
			 startcontainer[i] == p1.container);

			if (mask) {
				total.count++;
				total.sum += starteprice[i]*(100 - startdiscount[i]);
			}
		}
		
		
		total.count += sum_lanes_8(acc_counts);
		total.sum += sum_lanes_8(acc_total);
		return  total;
	};

	return parallel_reduce(blocked_range<size_t>(0, d.len, FLAGS_grain_size), init, body, addq19);
}

void viz_example(const lineitem_parts &, int * ) {
	// container, brand are categorical
	// quantity, eprice.
	// select avg(quantity), avg(eprice) from table group by container;
	// select avg(quantity), avg(eprice) from table (where rowid is selected) group by container;
	// select avg(quantity), avg(eprice) from table group by brand;
	// select avg(quantity), avg(eprice) from table (where rowid is selected) group by brand;
}
