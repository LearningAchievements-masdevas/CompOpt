all: build

.PHONY: build
build: src/logreg.hpp src/logreg.cpp src/data_gen.hpp src/main.cpp src/verbose.cpp src/verbose.hpp src/structures.hpp
	icx -I/home/masdevas/oneapi/tbb/latest/include -I/home/masdevas/oneapi/mkl/latest/include -I/home/masdevas/oneapi/advisor/latest/include \
		-lstdc++ \
		-L/home/masdevas/oneapi/tbb/latest/lib/intel64/gcc4.8 -ltbb \
		-L/home/masdevas/oneapi/mkl/latest/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -lpthread \
		-ldl src/*.cpp /home/masdevas/oneapi/advisor/2022.0.0/lib64/libittnotify.a \
		-O2 -std=c++17 -o exec
