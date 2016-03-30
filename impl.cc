#include "common.h"
#include <tbb/tbb.h>
#include <gflags/gflags.h>
#include "impl_helper.h"
#include "mask_table.h"

DEFINE_int32(grain_size, (1 << 14)/sizeof(data_t), "minimum amount of work (num array elts)");

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
	((($d).brand[($i)] == ($p).brand) $AND1 \
   (($d).container[($i)] == ($p).container) $AND2			\
	 (($d).quantity[($i)] < ($p).max_quantity) $AND3	\
	 (($d).quantity[($i)] >= ($p).min_quantity)) 


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
	auto body = 	[&](const auto & range, const auto & init)  {
		auto len = range.end() - range.begin();
		auto startbrand = &d.brand[range.begin()];
		auto startcontainer = &d.container[range.begin()];
		auto startquantity  = &d.quantity[range.begin()];
		auto starteprice  = &d.eprice[range.begin()];
		auto startdiscount  = &d.discount[range.begin()];

		Vec4qVec<k_acc_size> acc_counts(0);
		Vec4qVec<k_acc_size> acc_total(0);
		q19res total = init;

		constexpr auto modulus = ((uint64_t)std::numeric_limits<data_t>::max()) + 1;
		
		
		__declspec(align(64)) uint32_t buf[k_elts_per_buf] {};
		
		auto process_buffer = [&] (auto buf_size) {
#pragma forceinline recursive
			for (int idx = 0; idx < buf_size; idx += k_elts_per_vec ) {
				auto indices =  &buf[idx];

				auto containerv = gather(indices, startcontainer);
				auto quantityv = gather(indices, startquantity);
				auto epricev = gather(indices, starteprice);
				auto discountv = gather(indices, startdiscount);

				vec_t mask = (containerv == p1.container) & (quantityv >= p1.min_quantity)
					& (quantityv < p1.max_quantity);

				
				acc_counts += extend(mask & 1);
				acc_total += extend(mask & ((100 - discountv) * epricev));
			}
		};

		int j = 0;
		vec_t currbrands;
		const auto k_watermark = k_elts_per_buf - k_elts_per_vec;
		
#pragma forceinline recursive
		for (uint32_t i = 0; i < (range.end() - range.begin()); i+= k_elts_per_vec) {
			currbrands.load_a(&startbrand[i]);
			auto quals = currbrands == p1.brand;
			auto bigmask = _mm256_movemask_ps(_mm256_castsi256_ps(quals));

			for (int sub = 0; sub < 4; ++sub){
				auto charmask = (bigmask >> sub*8) & 0xff; // pick the lowest 8 bits
				auto delta_j = _mm_popcnt_u64(charmask);
				vec_t perm_mask;
				perm_mask.load_a(&mask_table[charmask]);
				auto store_pos = perm_mask + i + sub;
				_mm256_storeu_si256((__m256i*)&buf[j], store_pos);
				j+=delta_j;
			}

			if (j >= k_watermark) {
				process_buffer(k_watermark);
				vec_t last;
				last.load_a(&buf[k_watermark]);
				last.store_a(buf);
				j = j - k_watermark;
			}
		}
		
		const auto vec_tail_end = (j / k_elts_per_vec) * k_elts_per_vec ;
		process_buffer(vec_tail_end);

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
		
		
		total.count += acc_counts.sum_all();
		total.sum += acc_total.sum_all();
		return  total;
	};

	return parallel_reduce(blocked_range<size_t>(0, d.len, FLAGS_grain_size), init, body, addq19);
}



void q19lite_cluster (const lineitem_parts &d, q19params p1, lineitem_parts &output) {
	using namespace tbb;
	auto body = [&](const auto & range)  {
		auto len = range.end() - range.begin();
		
		auto brand = &d.brand[range.begin()];
		auto eprice  = &d.eprice[range.begin()];
		auto quantity = &d.quantity[range.begin()];
		auto container = &d.container[range.begin()];
		auto discount  = &d.discount[range.begin()];

		auto obrand = &output.brand[range.begin()];
		auto oeprice  = &output.eprice[range.begin()];
		auto oquantity = &output.quantity[range.begin()];
		auto ocontainer = &output.container[range.begin()];
		auto odiscount  = &output.discount[range.begin()];
		int output_j = 0;
		
		__declspec(align(64)) uint32_t buf_pos[k_elts_per_buf] {};

		__declspec(align(64)) data_t buf_brand_yes[k_elts_per_buf] {};
		__declspec(align(64)) data_t buf_brand_no[k_elts_per_buf] {};

		__declspec(align(64)) data_t buf_eprice_yes[k_elts_per_buf] {};
		__declspec(align(64)) data_t buf_eprice_no[k_elts_per_buf] {};

		__declspec(align(64)) data_t buf_quantity_yes[k_elts_per_buf] {};
		__declspec(align(64)) data_t buf_quantity_no[k_elts_per_buf] {};

		__declspec(align(64)) data_t buf_container_yes[k_elts_per_buf] {};
		__declspec(align(64)) data_t buf_container_no[k_elts_per_buf] {};

		__declspec(align(64)) data_t buf_discount_yes[k_elts_per_buf] {};
		__declspec(align(64)) data_t buf_discount_no[k_elts_per_buf] {};

		
		auto flush_buffer = [&] (auto buf_size, bool yes) {
			auto buf_brand = buf_brand_yes;
			auto buf_eprice = buf_eprice_yes;
			auto buf_quantity = buf_quantity_yes;
			auto buf_container = buf_container_yes;
			auto buf_discount = buf_discount_yes;
			
			if(!yes){
				buf_brand = buf_brand_no;
				buf_eprice = buf_eprice_no;
				buf_quantity = buf_quantity_no;
				buf_container = buf_container_no;
				buf_discount = buf_discount_no;
			}

			for (int i = 0; i < buf_size; i += 1 ) {
					obrand[output_j] = buf_brand[i];
					oeprice[output_j] = buf_eprice[i];
					oquantity[output_j] = buf_quantity[i];
					ocontainer[output_j] = buf_container[i];
					odiscount[output_j] = buf_discount[i];					
					output_j ++;
			}
		};

		int j = 0;
		int neg_j = 0;
		for (uint32_t i = 0; i < (range.end() - range.begin()); ++i) {
			if (brand[i] == p1.brand){
				buf_brand_yes[j] = brand[i];
				buf_eprice_yes[j] = eprice[i];
				buf_quantity_yes[j] = quantity[i];
				buf_container_yes[j] = container[i];
				buf_discount_yes[j] = discount[i];
				++j;
			} else {
				buf_brand_no[neg_j] = brand[i];
				buf_eprice_no[neg_j] = eprice[i];
				buf_quantity_no[neg_j] = quantity[i];
				buf_container_no[neg_j] = container[i];
				buf_discount_no[neg_j] = discount[i];					
				++neg_j;
			}
			
			if (j == k_elts_per_buf) { // modulo
				flush_buffer(k_elts_per_buf, true);
				j = 0;
			}

			if (neg_j == k_elts_per_buf) {
				flush_buffer(k_elts_per_buf, false);
				neg_j = 0;
			}
		}
		
		flush_buffer(j, true);
		flush_buffer(neg_j, false);
	};

	parallel_for(blocked_range<size_t>(0, d.len, FLAGS_grain_size), body);
}


void viz_example(const lineitem_parts &, int * ) {
	// container, brand are categorical
	// quantity, eprice.
	// select avg(quantity), avg(eprice) from table group by container;
	// select avg(quantity), avg(eprice) from table (where rowid is selected) group by container;
	// select avg(quantity), avg(eprice) from table group by brand;
	// select avg(quantity), avg(eprice) from table (where rowid is selected) group by brand;
}
