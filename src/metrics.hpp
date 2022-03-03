#ifndef METRICS_HPP
#define METRICS_HPP

#include <cmath>

#include "structures.hpp"

namespace metrics {

template <typename Container>
bool check_containers_eq(const Container& lhs, const Container& rhs, double eps) {
	if (lhs.size() != rhs.size()) {
		return false;
	}
	for (size_t index = 0; index < lhs.size(); ++index) {
		if (std::abs(lhs[index] - rhs[index]) >= eps) {
			return false;
		}
	}
	return true;
}

template <typename FPType>
bool check_forward_equality(const ForwardResult<FPType>& lhs, const ForwardResult<FPType>& rhs) {
	constexpr double eps = 1e-1;
	return check_containers_eq(lhs.sigm, rhs.sigm, eps) && (std::abs(lhs.logloss - rhs.logloss) < eps);
}

template <typename FPType>
bool check_gradient_equality(const GradientResult<FPType>& lhs, const GradientResult<FPType>& rhs) {
	constexpr double eps = 1;
	return check_containers_eq(lhs.weights_gradient, rhs.weights_gradient, eps) && (std::abs(lhs.beta_gradient - rhs.beta_gradient) < eps);
}

}

#endif
