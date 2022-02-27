#ifndef DATA_GEN_HPP
#define DATA_GEN_HPP

#include <random>
#include <tuple>

template <typename FPType>
std::tuple<std::vector<FPType>, std::vector<FPType>, FPType, std::vector<FPType>> generate_data(const Meta& meta) {
	std::vector<FPType> data(meta.rows_count * meta.columns_count), weights(meta.columns_count), groundTruth(meta.rows_count);
	FPType beta;

	std::uniform_real_distribution<FPType> uniform_dist(0, 100), ground_truth_dist(0, 1);
	std::random_device rd;
	for (auto& value : data) {
		value = uniform_dist(rd);
	}
	for (auto& value : weights) {
		value = uniform_dist(rd);
	}
	for (auto& value : groundTruth) {
		value = ground_truth_dist(rd);
		if (value < 0.5) {
			value = 0.;
		} else {
			value = 1.;
		}
	}
	beta = uniform_dist(rd);
	return std::make_tuple(data, weights, beta, groundTruth);
}

#endif
