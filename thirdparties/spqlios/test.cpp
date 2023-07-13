#include "fft_processor_spqlios.h"
#include <vector>
#include <iostream>

typedef std::vector<double> DoubleVec;
typedef std::vector<uint32_t> IntVec;

template<class T>
void put(const std::vector<T>& v)
{
	for (size_t i = 0, n = v.size(); i < n; i++) {
		std::cout << v[i] << ' ';
		if ((i % 16) == 15) std::cout << '\n';
	}
	std::cout << '\n';
}

void test(int n)
{
	FFT_Processor_Spqlios fft(n);
	DoubleVec a(n);
	for (int i = 0; i < n; i++) {
		a[i] = i * 1.1;
	}
	IntVec res(n);
	fft.execute_direct_torus32(res.data(), a.data());
	put(res);
}
int main()
{
#if 1
	test(128);
#else
	test(1024);
	test(2048);
#endif
}
