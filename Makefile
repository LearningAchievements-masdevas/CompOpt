all: build

.PHONY: build
build: src/*
	icx -I/home/masdevas/oneapi/tbb/latest/include -I/home/masdevas/oneapi/mkl/latest/include -I/home/masdevas/oneapi/advisor/latest/include \
		-lstdc++ -xCORE-AVX2 \
		-L/home/masdevas/oneapi/tbb/latest/lib/intel64/gcc4.8 -ltbb \
		-L/home/masdevas/oneapi/mkl/latest/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm \
		-ldl src/*.cpp /home/masdevas/oneapi/vtune/latest/lib64/libittnotify.a \
		-O2 -std=c++17 -o exec

.PHONY: clean
clean: exec data.dat extra.dat
	rm -rf exec data.dat extra.dat r00*
