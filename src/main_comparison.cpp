#include <chrono>

#include <mkl.h>

#include "logreg.hpp"
#include "structures.hpp"
#include "data_gen.hpp"

using FPType = float;

int main() {
	mkl_set_num_threads(1);
	Meta meta {
		static_cast<size_t>(1.5 * 1024 * 1024), // Cache size
		100, // Columns number
		100000 // Rows number
	};
	
	auto [data, weights, beta, groundTruth] = generate_data<FPType>(meta);

	auto result_forward = logreg_noopt::forward<FPType>(meta, data, weights, groundTruth, beta);
	auto result_gradient = logreg_noopt::gradient<FPType>(meta, data, weights, groundTruth, beta, result_forward);


	return 0;
}
