#include "mkl.h"
#include "logreg.hpp"

template <>
void call_gemv<float>(const CBLAS_TRANSPOSE trans, const MKL_INT m, const MKL_INT n, const float alpha, const float *a, const MKL_INT lda, const float *x, const float beta, float *y) {
	MKL_INT incx = 1;
	MKL_INT incy = 1;
	cblas_sgemv(CBLAS_LAYOUT::CblasRowMajor, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
