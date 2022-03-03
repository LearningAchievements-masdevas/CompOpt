#include <chrono>

#include <ittnotify.h>
#include <mkl.h>
#include <tbb/tbb.h>

#include "data_gen.hpp"
#include "logreg.hpp"
#include "metrics.hpp"
#include "structures.hpp"
#include "verbose.hpp"

__itt_domain *domain = __itt_domain_create("LogReg.Domain.Global");
__itt_string_handle *handle_data_gen =
    __itt_string_handle_create("Data Generation");
__itt_string_handle *handle_noopt = __itt_string_handle_create("Not Optimal");
__itt_string_handle *handle_opt = __itt_string_handle_create("Optimal");

using FPType = float;

int main() {
  tbb::task_arena arena(5);
  arena.execute([] {
    mkl_set_num_threads(1);
    Meta meta{
        static_cast<size_t>(1.5 * 1024 * 1024), // Cache size
        100,                                    // Columns number
        10000000                                // Rows number
    };
    bool verbosity = check_verbosity();

    __itt_task_begin(domain, __itt_null, __itt_null, handle_data_gen);
    verbose_print(verbosity, "Rows: ", meta.rows_count,
                  ", Columns: ", meta.columns_count);
    verbose_print(verbosity, "Number of threads: ",
                  tbb::this_task_arena::max_concurrency());
    verbose_print(verbosity, "# Start data generation");
    auto [data, weights, beta, groundTruth] =
        generate_data<FPType>(meta, "data.dat");
    __itt_task_end(domain);

    auto [data_extra, weights_extra, beta_extra, groundTruth_extra] =
        generate_data<FPType>(meta, "extra.dat");

    constexpr size_t real_runs = 100;

    verbose_print(verbosity, "# Start no optimal solution");
    auto start_noopt = std::chrono::system_clock::now();
    __itt_task_begin(domain, __itt_null, __itt_null, handle_noopt);
    for (size_t index = 0; index < real_runs; ++index) {
      auto result_forward_noopt = logreg_noopt::forward<FPType>(
          meta, data, weights, groundTruth, beta, verbosity);
      auto result_gradient_noopt =
          logreg_noopt::gradient<FPType>(meta, data, weights, groundTruth, beta,
                                         result_forward_noopt, verbosity);
    }
    __itt_task_end(domain);
    auto finish_noopt = std::chrono::system_clock::now();
    verbose_print(verbosity, "# No optimal solution finished");
    std::cout << "No opt time (sec): "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     finish_noopt - start_noopt)
                         .count() /
                     1e6 / real_runs
              << std::endl;
    auto result_forward_noopt = logreg_noopt::forward<FPType>(
        meta, data, weights, groundTruth, beta, verbosity);
    auto result_gradient_noopt =
        logreg_noopt::gradient<FPType>(meta, data, weights, groundTruth, beta,
                                       result_forward_noopt, verbosity);

    std::this_thread::sleep_for(std::chrono::seconds(3));

    constexpr size_t extra_runs = 10;
    for (size_t run_index = 0; run_index < extra_runs; ++run_index) {
      auto result_forward_extra =
          logreg_noopt::forward<FPType>(meta, data_extra, weights_extra,
                                        groundTruth_extra, beta_extra, false);
      logreg_noopt::gradient<FPType>(meta, data_extra, weights_extra,
                                     groundTruth_extra, beta_extra,
                                     result_forward_extra, false);
    }

    std::this_thread::sleep_for(std::chrono::seconds(3));

    verbose_print(verbosity, "# Start optimal solution");
    auto start_opt = std::chrono::system_clock::now();
    __itt_task_begin(domain, __itt_null, __itt_null, handle_opt);
    for (size_t index = 0; index < real_runs; ++index) {
      auto [result_forward_opt, result_gradient_opt] =
          logreg_opt::forward_and_gradient<FPType>(
              meta, data, weights, groundTruth, beta, verbosity);
    }
    __itt_task_end(domain);
    auto finish_opt = std::chrono::system_clock::now();
    verbose_print(verbosity, "# Optimal solution finished");
    std::cout << "Opt time (sec): "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     finish_opt - start_opt)
                         .count() /
                     1e6 / real_runs
              << std::endl;
    auto [result_forward_opt, result_gradient_opt] =
        logreg_opt::forward_and_gradient<FPType>(meta, data, weights,
                                                 groundTruth, beta, verbosity);

    bool forward_equality = metrics::check_forward_equality(
        result_forward_noopt, result_forward_opt);
    if (!forward_equality) {
      verbose_print(true, "!!! Forward results are not equal");
    } else {
      verbose_print(true, "# Forward results are equal");
    }
    bool gradient_equality = metrics::check_gradient_equality(
        result_gradient_noopt, result_gradient_opt);
    if (!gradient_equality) {
      verbose_print(true, "!!! Gradient results are not equal");
    } else {
      verbose_print(true, "# Gradient results are equal");
    }
    verbose_print(verbosity, "# Finished!");
  });
  return 0;
}
