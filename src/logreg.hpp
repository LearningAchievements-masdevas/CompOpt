#ifndef LOGREG_HPP
#define LOGREG_HPP

#include <cmath>

#include <tbb/tbb.h>
#include <mkl.h>

#include "structures.hpp"

template <typename FPType>
void call_gemv(const CBLAS_TRANSPOSE trans, const MKL_INT m, const MKL_INT n, const FPType alpha, const FPType *a, const MKL_INT lda, const FPType *x, const FPType beta, FPType *y);

template <typename FPType>
inline FPType sigmoid(FPType value) {
	return 1. / (1. + std::exp(-value));
}

namespace logreg_noopt {

// Dimensions note. Data: rows x cols, weights: cols
template <typename FPType>
ForwardResult<FPType> forward(const Meta& meta, const std::vector<FPType>& data, const std::vector<FPType>& weights, const std::vector<float>& groundTruth, const FPType beta) {
	ForwardResult<FPType> result(meta.rows_count);
	const size_t rows_in_block = meta.l2_cache_size * 0.8 / (meta.columns_count * sizeof(FPType));
	const size_t blocks_count = meta.rows_count / rows_in_block;
	using TLS = tbb::enumerable_thread_specific<FPType>;
	TLS tls(FPType{0.});

	// Calculate sigmoid(Ax+b)
	tbb::parallel_for(tbb::blocked_range<int>(0, blocks_count),
		[&](tbb::blocked_range<int> r) {
		typename TLS::reference local_tls = tls.local();
	    for (int block_index = r.begin(); block_index < r.end(); ++block_index) {
	    	const FPType* data_ptr = data.data() + rows_in_block * meta.columns_count * block_index;
	    	const size_t rows_to_process = block_index + 1 == blocks_count ? meta.rows_count - rows_in_block * block_index : rows_in_block;
	    	constexpr CBLAS_TRANSPOSE trans = CBLAS_TRANSPOSE::CblasNoTrans;
    		constexpr FPType alpha = 1.;
			const MKL_INT lda = meta.columns_count;
			constexpr FPType beta = 0.;
	    	call_gemv<FPType>(trans, rows_to_process, meta.columns_count, alpha, const_cast<FPType*>(data_ptr), lda, const_cast<FPType*>(weights.data()), beta, const_cast<FPType*>(result.sigm.data()));
	    	size_t start_row = rows_in_block * block_index;
	    	for (size_t row_index = 0; row_index < rows_to_process; ++row_index) {
	    		const size_t local_row = start_row + row_index;
	    		result.sigm[local_row] += beta;
	    		result.sigm[local_row] = sigmoid(result.sigm[local_row]);
	    		const FPType local_value = result.sigm[local_row];
	    		local_tls += (-groundTruth[local_row] * std::log(local_value) - (1 - groundTruth[local_row]) * std::log(1 - local_value));
	    	}
	    }
	}, tbb::static_partitioner());

	for (auto& value : tls) {
		result.logloss += value;
	}

	return result;
}

template <typename FPType>
GradientResult<FPType> gradient(const Meta& meta, const std::vector<FPType>& data, const std::vector<FPType>& weights, const std::vector<float>& groundTruth, const FPType beta, const ForwardResult<FPType>& forward_result) {
	GradientResult<FPType> result(meta.columns_count);
	const size_t rows_in_block = meta.l2_cache_size * 0.8 / (meta.columns_count * sizeof(FPType));
	const size_t blocks_count = meta.rows_count / rows_in_block;

	using TLS = tbb::enumerable_thread_specific<std::tuple<std::vector<FPType>, std::vector<FPType>, FPType>>;
	TLS tls_local_grad(std::make_tuple(std::vector<FPType>(meta.columns_count), std::vector<FPType>(rows_in_block), static_cast<FPType>(0)));
	tbb::parallel_for(tbb::blocked_range<int>(0, blocks_count),
		[&](tbb::blocked_range<int> r) {
		typename TLS::reference local_tls = tls_local_grad.local();
		auto& [local_grad, local_sigm_logloss_derivatives, local_beta] = local_tls;

		for (int block_index = r.begin(); block_index < r.end(); ++block_index) {
			const size_t start_row = rows_in_block * block_index;
			const size_t rows_to_process = block_index + 1 == blocks_count ? meta.rows_count - rows_in_block * block_index : rows_in_block;
			for (size_t index = 0; index < rows_to_process; ++index) {
				const auto abs_index = start_row + index;
				const auto& sigm = forward_result.sigm[abs_index];
				const auto& gt = groundTruth[abs_index];
				local_sigm_logloss_derivatives[index] = 
					sigm * (1 - sigm) * // Sigm derivative
					(-gt / sigm - (1 - gt) / (1 - sigm)); // LogLoss derivative
				local_beta += local_sigm_logloss_derivatives[index];
			}
			const FPType* data_ptr = data.data() + rows_in_block * meta.columns_count * block_index;
			const FPType* local_sigm_logloss_derivatives = forward_result.sigm.data();
			constexpr CBLAS_TRANSPOSE trans = CBLAS_TRANSPOSE::CblasTrans;
    		constexpr FPType alpha = 1.;
			const MKL_INT lda = meta.columns_count;
			constexpr FPType beta = 1.;
	    	call_gemv<FPType>(trans, meta.columns_count, rows_to_process, alpha, const_cast<FPType*>(data_ptr), lda, const_cast<FPType*>(local_sigm_logloss_derivatives), beta, local_grad.data());
		}
	}, tbb::static_partitioner());


	for (auto& [local_grad, local_sigm_logloss_derivatives, local_beta] : tls_local_grad) {
		for (size_t index = 0; index < meta.columns_count; ++index) {
			result.weights_gradient[index] += local_grad[index];
		}
		result.beta_gradient += local_beta;
	}
	return result;
}

}

#endif
