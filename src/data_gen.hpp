#ifndef DATA_GEN_HPP
#define DATA_GEN_HPP

#include <random>
#include <tuple>

#include <tbb/tbb.h>

template <typename FPType>
std::tuple<std::vector<FPType>, std::vector<FPType>, FPType, std::vector<FPType>> generate_data(const Meta& meta) {
	std::vector<FPType> data(meta.rows_count * meta.columns_count), weights(meta.columns_count), groundTruth(meta.rows_count);
	FPType beta;

	std::uniform_real_distribution<FPType> uniform_dist(0, 100), ground_truth_dist(0, 1);
	std::random_device rd;
	const size_t rows_in_block = meta.l2_cache_size * 0.8 / (meta.columns_count * sizeof(FPType));
	const size_t blocks_count = meta.rows_count / rows_in_block + !!(meta.rows_count % rows_in_block);
	tbb::parallel_for(tbb::blocked_range<int>(0, blocks_count),
		[&](tbb::blocked_range<int> r) {
		for (int block_index = r.begin(); block_index < r.end(); ++block_index) {
			const size_t start_row = rows_in_block * block_index;
			const size_t rows_to_process = block_index + 1 == blocks_count ? meta.rows_count - rows_in_block * block_index : rows_in_block;
			for (size_t index = start_row * meta.columns_count; index < (start_row + rows_to_process) * meta.columns_count; ++index) {
				data[index] = uniform_dist(rd);
			}
			for (size_t index = start_row; index < start_row + rows_to_process; ++index) {
				FPType value = ground_truth_dist(rd);
				if (value < 0.5) {
					value = 0.;
				} else {
					value = 1.;
				}
				groundTruth[index] = value;
			}
		}
	}, tbb::static_partitioner());

	for (auto& value : weights) {
		value = uniform_dist(rd);
	}
	
	beta = uniform_dist(rd);
	return std::make_tuple(data, weights, beta, groundTruth);
}

#endif
