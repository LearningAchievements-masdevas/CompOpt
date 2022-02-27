all: build

.PHONY: build
build: src/logreg.hpp src/logreg.cpp src/data_gen.hpp src/main_comparison.cpp src/structures.hpp
	icx -I/home/masdevas/oneapi/tbb/latest/include -I/home/masdevas/oneapi/mkl/latest/include \
		-lstdc++ \
		-L/home/masdevas/oneapi/tbb/2021.5.1/lib/intel64/gcc4.8 -ltbb \
		-L/home/masdevas/oneapi/mkl/2022.0.2/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -lpthread \
		-ldl src/*.cpp -O2 -std=c++17
