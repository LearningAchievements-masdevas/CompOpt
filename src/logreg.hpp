#ifndef LOGREG_HPP
#define LOGREG_HPP

#include <cmath>

#include <tbb/tbb.h>
#include <mkl.h>

#include "structures.hpp"
#include "verbose.hpp"

constexpr double eps = 1e-7;

template <typename FPType>
void call_gemv(const CBLAS_TRANSPOSE trans, const MKL_INT m, const MKL_INT n, const FPType alpha, const FPType *a, const MKL_INT lda, const FPType *x, const FPType beta, FPType *y);

template <typename FPType>
inline FPType sigmoid(FPType value) {
	return 1. / (1. + std::exp(-value));
}

namespace logreg_noopt {

// Dimensions note. Data: rows x cols, weights: cols
template <typename FPType>
ForwardResult<FPType> forward(const Meta& meta, const std::vector<FPType>& data, const std::vector<FPType>& weights, const std::vector<float>& groundTruth, const FPType beta_weight, const bool verbosity) {
	ForwardResult<FPType> result(meta.rows_count);
	const size_t rows_in_block = meta.l2_cache_size * 0.8 / (meta.columns_count * sizeof(FPType));
	const size_t blocks_count = meta.rows_count / rows_in_block + !!(meta.rows_count % rows_in_block);
	using TLS = tbb::enumerable_thread_specific<FPType>;
	TLS tls(static_cast<FPType>(0.));

	tbb::parallel_for(tbb::blocked_range<int>(0, blocks_count),
		[&](tbb::blocked_range<int> r) {
		typename TLS::reference local_logloss = tls.local();
	    for (int block_index = r.begin(); block_index < r.end(); ++block_index) {
	    	const size_t start_row = rows_in_block * block_index;
	    	const FPType* data_ptr = data.data() + start_row * meta.columns_count;
	    	FPType* result_ptr = result.sigm.data() + start_row;
	    	const FPType* gt_ptr = groundTruth.data() + start_row;
	    	const size_t rows_to_process = block_index + 1 == blocks_count ? meta.rows_count - rows_in_block * block_index : rows_in_block;

	    	constexpr CBLAS_TRANSPOSE trans = CBLAS_TRANSPOSE::CblasNoTrans;
    		constexpr FPType alpha = 1.;
			const MKL_INT lda = meta.columns_count;
			constexpr FPType beta = 0.;
	    	call_gemv<FPType>(trans, rows_to_process, meta.columns_count, alpha, data_ptr, lda, weights.data(), beta, result_ptr);

	    	for (size_t row_index = 0; row_index < rows_to_process; ++row_index) {
	    		result_ptr[row_index] += beta_weight;
	    		result_ptr[row_index] = sigmoid(result_ptr[row_index]);
	    		const FPType local_value = result_ptr[row_index];
	    		local_logloss += (-gt_ptr[row_index] * std::log(local_value + eps) - (1 - gt_ptr[row_index]) * std::log(1 - local_value + eps));
	    	}
	    }
	}, tbb::static_partitioner());

	// verbose_print(verbosity, "# Forward. Start reduction");
	for (auto& value : tls) {
		result.logloss += value;
	}
	// verbose_print(verbosity, "# Forward. Finished");
	return result;
}

template <typename FPType>
GradientResult<FPType> gradient(const Meta& meta, const std::vector<FPType>& data, const std::vector<FPType>& weights, const std::vector<float>& groundTruth, const FPType beta, const ForwardResult<FPType>& forward_result, bool verbosity) {
	GradientResult<FPType> result(meta.columns_count);
	const size_t rows_in_block = meta.l2_cache_size * 0.8 / (meta.columns_count * sizeof(FPType));
	const size_t blocks_count = meta.rows_count / rows_in_block + !!(meta.rows_count % rows_in_block);

	using TLS = tbb::enumerable_thread_specific<std::tuple<std::vector<FPType>, std::vector<FPType>, FPType>>;
	TLS tls_local_grad(std::make_tuple(std::vector<FPType>(meta.columns_count), std::vector<FPType>(rows_in_block), static_cast<FPType>(0)));
	tbb::parallel_for(tbb::blocked_range<int>(0, blocks_count),
		[&](tbb::blocked_range<int> r) {
		typename TLS::reference local_tls = tls_local_grad.local();
		auto& [local_grad, local_sigm_logloss_derivatives, local_beta] = local_tls;

		for (int block_index = r.begin(); block_index < r.end(); ++block_index) {
			const size_t start_row = rows_in_block * block_index;
			const size_t rows_to_process = block_index + 1 == blocks_count ? meta.rows_count - rows_in_block * block_index : rows_in_block;
			const FPType* data_ptr = data.data() + start_row * meta.columns_count;
			for (size_t index = 0; index < rows_to_process; ++index) {
				const auto abs_index = start_row + index;
				const auto& sigm = forward_result.sigm[abs_index];
				const auto& gt = groundTruth[abs_index];
				local_sigm_logloss_derivatives[index] = 
					sigm * (1 - sigm) * // Sigm derivative
					(-gt / sigm - (1 - gt) / (1 - sigm)); // LogLoss derivative
				local_beta += local_sigm_logloss_derivatives[index];
			}
			
			constexpr CBLAS_TRANSPOSE trans = CBLAS_TRANSPOSE::CblasTrans;
    		constexpr FPType alpha = 1.;
			const MKL_INT lda = meta.columns_count;
			constexpr FPType beta = 1.;
	    	call_gemv<FPType>(trans, rows_to_process, meta.columns_count, alpha, data_ptr, lda, local_sigm_logloss_derivatives.data(), beta, local_grad.data());
		}
	}, tbb::static_partitioner());

	// verbose_print(verbosity, "# Backward. Start reduction");
	for (auto& [local_grad, local_sigm_logloss_derivatives, local_beta] : tls_local_grad) {
		for (size_t index = 0; index < meta.columns_count; ++index) {
			result.weights_gradient[index] += local_grad[index];
		}
		result.beta_gradient += local_beta;
	}
	// verbose_print(verbosity, "# Backward. Finished");
	return result;
}

}

namespace logreg_opt {

template <typename FPType>
std::pair<ForwardResult<FPType>, GradientResult<FPType>> forward_and_gradient(const Meta& meta, const std::vector<FPType> data, const std::vector<FPType>& weights, const std::vector<float>& groundTruth, const FPType beta_weight, const bool verbosity) {
	ForwardResult<FPType> result_forward(meta.rows_count);
	GradientResult<FPType> result_gradient(meta.columns_count);
	const size_t rows_in_block = meta.l2_cache_size * 0.8 / (meta.columns_count * sizeof(FPType));
	const size_t blocks_count = meta.rows_count / rows_in_block + !!(meta.rows_count % rows_in_block);
	using TLS = tbb::enumerable_thread_specific<std::tuple<std::vector<FPType>, std::vector<FPType>, FPType, FPType>>;
	TLS tls(std::make_tuple(std::vector<FPType>(meta.columns_count), std::vector<FPType>(rows_in_block), static_cast<FPType>(0.), static_cast<FPType>(.0)));

	// verbose_print(verbosity, "# Opt parallel section");
	tbb::parallel_for(tbb::blocked_range<int>(0, blocks_count),
		[&](tbb::blocked_range<int> r) {
		typename TLS::reference local_tls = tls.local();
		auto& [local_grad, local_sigm_logloss_derivatives, local_beta, local_logloss] = local_tls;
	    for (int block_index = r.begin(); block_index < r.end(); ++block_index) {

	    	// Calculate forward
	    	const size_t start_row = rows_in_block * block_index;
	    	const FPType* data_ptr = data.data() + start_row * meta.columns_count;
	    	FPType* result_ptr = result_forward.sigm.data() + start_row;
	    	const FPType* gt_ptr = groundTruth.data() + start_row;
	    	const size_t rows_to_process = block_index + 1 == blocks_count ? meta.rows_count - rows_in_block * block_index : rows_in_block;

	    	{
		    	constexpr CBLAS_TRANSPOSE trans = CBLAS_TRANSPOSE::CblasNoTrans;
	    		constexpr FPType alpha = 1.;
				const MKL_INT lda = meta.columns_count;
				constexpr FPType beta = 0.;
				call_gemv<FPType>(trans, rows_to_process, meta.columns_count, alpha, data_ptr, lda, weights.data(), beta, result_ptr);
			}

	    	for (size_t row_index = 0; row_index < rows_to_process; ++row_index) {
	    		result_ptr[row_index] += beta_weight;
	    		result_ptr[row_index] = sigmoid(result_ptr[row_index]);
	    		const FPType local_value = result_ptr[row_index];
	    		local_logloss += (-gt_ptr[row_index] * std::log(local_value + eps) - (1 - gt_ptr[row_index]) * std::log(1 - local_value + eps));
	    	}

	    	// Calculate gradient
			for (size_t index = 0; index < rows_to_process; ++index) {
				const auto& sigm = result_ptr[index];
				const auto& gt = gt_ptr[index];
				local_sigm_logloss_derivatives[index] = 
					sigm * (1 - sigm) * // Sigm derivative
					(-gt / sigm - (1 - gt) / (1 - sigm)); // LogLoss derivative
				local_beta += local_sigm_logloss_derivatives[index];
			}
			{
				constexpr CBLAS_TRANSPOSE trans = CBLAS_TRANSPOSE::CblasTrans;
	    		constexpr FPType alpha = 1.;
				const MKL_INT lda = meta.columns_count;
				constexpr FPType beta = 1.;
		    	call_gemv<FPType>(trans, rows_to_process, meta.columns_count, alpha, data_ptr, lda, local_sigm_logloss_derivatives.data(), beta, local_grad.data());
	    	}
	    }
	}, tbb::static_partitioner());

	// verbose_print(verbosity, "# Opt reduction");
	for (auto& [local_grad, local_sigm_logloss_derivatives, local_beta, local_logloss] : tls) {
		for (size_t index = 0; index < meta.columns_count; ++index) {
			result_gradient.weights_gradient[index] += local_grad[index];
		}
		result_gradient.beta_gradient += local_beta;
		result_forward.logloss += local_logloss;
	}

	return std::make_pair(result_forward, result_gradient);
}

}

#endif
