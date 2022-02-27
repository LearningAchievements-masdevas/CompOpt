#include <chrono>

#include <mkl.h>
#include <ittnotify.h>
#include <tbb/tbb.h>

#include "logreg.hpp"
#include "structures.hpp"
#include "data_gen.hpp"
#include "verbose.hpp"

__itt_domain* domain = __itt_domain_create("LogReg.Domain.Global");
__itt_string_handle* handle_data_gen = __itt_string_handle_create("Data Generation");
__itt_string_handle* handle_noopt = __itt_string_handle_create("Not Optimal");
__itt_string_handle* handle_opt = __itt_string_handle_create("Optimal");

using FPType = float;

int main() {
	tbb::task_arena arena;
	arena.execute([]{
		mkl_set_num_threads(1);
		Meta meta {
			static_cast<size_t>(1.5 * 1024 * 1024), // Cache size
			100, // Columns number
			100000 // Rows number
		};
		bool verbosity = check_verbosity();

		verbose_print(verbosity, "Rows: ", meta.rows_count, ", Columns: ", meta.columns_count);
		verbose_print(verbosity, "Number of threads: ", tbb::this_task_arena::max_concurrency());
		verbose_print(verbosity, "# Start data generation");
		__itt_task_begin(domain, __itt_null, __itt_null, handle_data_gen);
		auto [data, weights, beta, groundTruth] = generate_data<FPType>(meta);
		__itt_task_end(domain);

		verbose_print(verbosity, "# Start no optimal solution");
		__itt_task_begin(domain, __itt_null, __itt_null, handle_noopt);
		verbose_print(verbosity, "# Start no optimal solution: forward");
		auto result_forward = logreg_noopt::forward<FPType>(meta, data, weights, groundTruth, beta, verbosity);
		verbose_print(verbosity, "# Start no optimal solution: backward");
		auto result_gradient = logreg_noopt::gradient<FPType>(meta, data, weights, groundTruth, beta, result_forward, verbosity);
		__itt_task_end(domain);

		verbose_print(verbosity, "# Finished!");
	});
	return 0;
}
