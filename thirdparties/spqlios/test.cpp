#include "fft_processor_spqlios.h"
#include <vector>
#include <iostream>
#include <cybozu/benchmark.hpp>

typedef std::vector<double> DoubleVec;
typedef std::vector<uint32_t> IntVec;

template<class T>
void put(const std::vector<T>& v)
{
	for (size_t i = 0, n = v.size(); i < n; i++) {
		if ((i % 16) > 0) std::cout << ' ';
		std::cout << v[i];
		if ((i % 16) == 15) std::cout << '\n';
	}
	std::cout << '\n';
}

void test(int n)
{
	printf("n=%d\n", n);
	FFT_Processor_Spqlios fft(n);
	DoubleVec a(n);
	for (int i = 0; i < n; i++) {
		a[i] = sin(0.3*i) * i * 123.4;
	}
	IntVec res(n);
	fft.execute_direct_torus32(res.data(), a.data());
	CYBOZU_BENCH_C("fft", 10000, fft.execute_direct_torus32, res.data(), a.data());
	put(res);
}
int main()
{
	test(32);
	test(1024);
	test(2048);
}
