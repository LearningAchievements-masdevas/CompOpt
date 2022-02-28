#!/bin/bash
source /home/masdevas/oneapi/vtune/latest/vtune-vars.sh
source /home/masdevas/oneapi/compiler/2022.0.2/env/vars.sh
export INTEL_LIBITTNOTIFY64=/home/masdevas/oneapi/advisor/latest/lib64/runtime/libittnotify_collector.so
export LD_LIBRARY_PATH=/home/masdevas/oneapi/tbb/latest/lib/intel64/gcc4.8:/home/masdevas/oneapi/mkl/latest/lib/intel64:${LD_LIBRARY_PATH}
