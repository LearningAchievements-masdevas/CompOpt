#ifndef STRUCTURES_HPP
#define STRUCTURES_HPP

#include <vector>

template <typename FPType>
struct ForwardResult {
	ForwardResult(size_t size) : sigm(size), logloss(0.) {}

	std::vector<FPType> sigm;
	FPType logloss;
};

template <typename FPType>
struct GradientResult {
	GradientResult(size_t size) : weights_gradient(size), beta_gradient(0.) {}

	std::vector<FPType> weights_gradient;
	FPType beta_gradient;
};

struct Meta {
	size_t l2_cache_size;
	size_t columns_count;
	size_t rows_count;
};

#endif
