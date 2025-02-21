#pragma once

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>

namespace raintt {
template <typename T>
constexpr T ipow(T num, unsigned int pow)
{
    return (pow >= sizeof(unsigned int) * 8) ? 0
           : pow == 0                        ? 1
                                             : num * ipow(num, pow - 1);
}

using Word = uint32_t;
using SWord = int32_t;
using DoubleWord = uint64_t;
using DoubleSWord = int64_t;
constexpr uint k = 5;
constexpr uint radixbit = 3;
constexpr uint radixs2 = 1U << (radixbit - 1);
constexpr Word K = ipow<Word>(k, radixs2);
constexpr uint shiftunit = 5;
constexpr uint shiftamount = radixs2 * shiftunit;
constexpr SWord shiftval = 1 << shiftamount;
constexpr uint wordbits = 32;
constexpr Word wordmask = (1ULL << wordbits) - 1;
constexpr SWord P = (K << shiftamount) + 1;

constexpr SWord R = (1ULL << wordbits) % P;
constexpr SWord R2 = (static_cast<DoubleWord>(R) * R) % P;

// https://en.wikipedia.org/wiki/Montgomery_modular_multiplication
constexpr Word REDC(const DoubleWord T)
{
    const Word T0 = T & wordmask;
    const Word m = (((T0 * K) << shiftamount) - T0) & wordmask;
    const Word t =
        (T + ((static_cast<DoubleWord>(m) * K) << shiftamount) + m) >> wordbits;
    return (t > P) ? t - P : t;
}

// https://eprint.iacr.org/2018/039
SWord SREDC(const DoubleWord a)
{
    const Word a0 = a & wordmask;
    const SWord a1 = a >> wordbits;
    const SWord m = -((a0 * K) * shiftval) + a0;
    const SWord t1 =
        (((static_cast<DoubleSWord>(m) * K) << shiftamount) + m) >> wordbits;
    return a1 - t1;
}

SWord AddMod(const SWord a, const SWord b)
{
    SWord add = a + b;
    if (add >= P)
        return add - P;
    else if (add <= -P)
        return add + P;
    else
        return add;
}

SWord SubMod(const SWord a, const SWord b)
{
    SWord sub = a - b;
    if (sub >= P)
        return sub - P;
    else if (sub <= -P)
        return sub + P;
    else
        return sub;
}

constexpr Word MulREDC(const Word a, const Word b)
{
    const DoubleWord mul = static_cast<DoubleWord>(a) * b;
    return REDC(mul);
}

SWord MulSREDC(const SWord a, const SWord b)
{
    const DoubleSWord mul = static_cast<DoubleSWord>(a) * b;
    return SREDC(mul);
}

constexpr Word PowREDC(const Word a, const uint e)
{
    Word res = 1;
    const Word aR = MulREDC(R2, a);
    for (uint i = 0; i < e; i++) res = MulREDC(res, aR);
    return res;
}

template <Word a, Word b>
constexpr Word ext_gcd(SWord &x, SWord &y)
{
    if constexpr (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    else {
        Word d = ext_gcd<b, a % b>(y, x);
        y -= a / b * x;
        return d;
    }
}

template <Word a>
constexpr Word inv_mod()
{
    SWord x, y;
    const Word g = ext_gcd<a, P>(x, y);
    if (g != 1) {
        throw "Inverse doesn't exist";
    }
    else {
        return (x % P + P) % P;  // this line ensures x is positive
    }
}

// NTT related
constexpr SWord W = PowREDC(11, K);

template <uint8_t bit>
uint32_t BitReverse(uint32_t in)
{
    if constexpr (bit > 1) {
        const uint32_t center = in & ((bit & 1) << (bit / 2));
        return (BitReverse<bit / 2>(in & ((1U << (bit / 2)) - 1))
                << (bit + 1) / 2) |
               center | BitReverse<bit / 2>(in >> ((bit + 1) / 2));
    }
    else {
        return in;
    }
}

template <uint Nbit,uint radixbit>
std::unique_ptr<std::array<std::array<SWord, 1U << Nbit>, 2>> TwistGen()
{
    constexpr uint N = 1U << Nbit;
    constexpr uint8_t remainder = ((Nbit - 1) % radixbit) + 1;
    const Word invN = inv_mod<N>();

    std::unique_ptr<std::array<std::array<SWord, 1U << Nbit>, 2>> twist =
        std::make_unique<std::array<std::array<SWord, 1U << Nbit>, 2>>();
    const Word wR = MulREDC(PowREDC(W, 1U << (shiftamount - Nbit - 1)), R2);
    (*twist)[1][0] = R;
    for (uint i = 1; i < N; i++)
        (*twist)[1][i] = MulREDC((*twist)[1][i - 1], wR);
    // assert(((*twist)[1][N - 1] * w)== 1);

    (*twist)[0][N - 1] = MulREDC(MulREDC((*twist)[1][N - 1], wR),
                                 static_cast<DoubleWord>(invN) * wR % P);
    (*twist)[0][0] = (static_cast<DoubleWord>(invN) * R) % P;
    for (uint32_t i = 2; i < N; i++)
        (*twist)[0][N - i] = MulREDC((*twist)[0][N - i + 1], wR);
    assert(MulREDC((*twist)[0][1], wR) == (*twist)[0][0]);

    if constexpr(remainder!=1) for (uint i = 0; i < N; i++) (*twist)[1][i] = MulREDC((*twist)[1][i], R2);
    return twist;
}

template <uint32_t Nbit>
inline std::unique_ptr<std::array<std::array<std::array<SWord, 1U << Nbit>, 2>, 2>> TableGen()
{
    constexpr uint32_t N = 1U << Nbit;

    std::unique_ptr<std::array<std::array<std::array<SWord, N>, 2>, 2>> table =
        std::make_unique<std::array<std::array<std::array<SWord, N>, 2>, 2>>();
    const Word w = PowREDC(W, 1ULL << (shiftamount - Nbit));
    const Word wR = MulREDC(w, R2);
    (*table)[0][0][0] = (*table)[1][0][0] = R;
    (*table)[0][1][0] = (*table)[1][1][0] = R2;
    for (uint32_t i = 1; i < N; i++)
        (*table)[1][0][i] = MulREDC((*table)[1][0][i - 1], wR);
    assert(MulREDC((*table)[1][0][N - 1], wR) == R);
    for (uint32_t i = 1; i < N; i++)
        (*table)[1][1][i] = MulREDC((*table)[1][0][i], R2);
    for(int j = 0; j<2; j++) for (uint32_t i = 1; i < N; i++)(*table)[0][j][i] = (*table)[1][j][N - i];
    return table;
}

void ButterflyAddBothMod(DoubleSWord *const res, const uint size)
{
    for (uint index = 0; index < size / 2; index++) {
        const SWord temp = res[index];
        res[index] = AddMod(res[index], res[index + size / 2]);
        res[index + size / 2] = SubMod(temp, res[index + size / 2]);
    }
}

void ButterflyAddAddMod(DoubleSWord *const res, const uint size)
{
    for (uint index = 0; index < size / 2; index++) {
        const SWord temp = res[index];
        res[index] = AddMod(res[index], res[index + size / 2]);
        res[index + size / 2] = temp - res[index + size / 2];
    }
}

void ButterflyAdd(DoubleSWord *const res, const uint size)
{
    for (uint index = 0; index < size / 2; index++) {
        const SWord temp = res[index];
        res[index] += res[index + size / 2];
        res[index + size / 2] = temp - res[index + size / 2];
    }
}

void ButterflyAddBothSREDC(DoubleSWord *const res, const uint size)
{
    for (uint index = 0; index < size / 2; index++) {
        const DoubleSWord temp = res[index];
        res[index] = SREDC(res[index]+res[index + size / 2]);
        res[index + size / 2] = SREDC(temp - res[index + size / 2]);
    }
}

template <uint32_t Nbit>
inline void TwiddleMul(DoubleSWord *const res, const uint size, const uint stride,
                       const std::array<SWord, 1 << Nbit> &table)
{
    for (uint32_t index = 0; index < size; index++)
        res[index] = MulSREDC(res[index], table[stride * index]);
}

template <uint8_t radixbit, bool last>
inline void INTTradixButterfly(DoubleSWord *const res, const uint32_t size)
{
    static_assert(radixbit <= 2, "radix 4 is the maximum!");
    if constexpr (radixbit == 1) {
        ButterflyAddBothMod(res, size);
    }else if constexpr(radixbit == 2){
        if constexpr(!last){
            ButterflyAddAddMod(res, size);
            ButterflyAddBothMod(&res[0], size / 2);
        }else{
            ButterflyAdd(res, size);
            ButterflyAddBothSREDC(&res[0], size / 2);
        }
        const uint32_t block = size >> radixbit;
        for (int i = 1; i < (1 << (radixbit - 1)); i++)
            for (int j = 0; j < block; j++)
                res[i * block + j + size / 2] = (res[i * block + j + size / 2] * ipow<DoubleSWord>(k,i*(radixs2>>(radixbit-1))))<<(shiftamount>>(radixbit-1));
        ButterflyAddBothSREDC(&res[size / 2], size / 2);
    }
}

template <uint32_t Nbit, uint8_t radixbit>
inline void INTTradix(DoubleSWord *const res, const uint32_t size,
                      const uint32_t num_block,
                      const std::array<std::array<SWord, 1 << Nbit>,2> &table)
{
    INTTradixButterfly<radixbit,false>(res, size);
    if constexpr(radixbit==1){
        for (uint i = 1; i < (1 << radixbit); i++)
            TwiddleMul<Nbit>(&res[i * (size >> radixbit)], size >> radixbit,
                            BitReverse<radixbit>(i) * num_block, table[0]);
    }else{
        TwiddleMul<Nbit>(&res[1 * (size >> radixbit)], size >> radixbit,
                        BitReverse<radixbit>(1) * num_block, table[0]);
        for (uint i = 2; i < (1 << radixbit); i++)
            TwiddleMul<Nbit>(&res[i * (size >> radixbit)], size >> radixbit,
                            BitReverse<radixbit>(i) * num_block, table[1]);
    }
}

template <uint32_t Nbit, uint8_t radixbit>
inline void INTT(std::array<DoubleSWord, 1 << Nbit> &res,
                 const std::array<std::array<SWord, 1 << Nbit>,2> &table)
{
    for (uint8_t sizebit = Nbit; sizebit > radixbit; sizebit -= radixbit) {
        const uint32_t size = 1U << sizebit;
        const uint32_t num_block = 1U << (Nbit - sizebit);
        for (uint32_t block = 0; block < num_block; block++)
            INTTradix<Nbit, radixbit>(&res[size * block], size, num_block,
                                      table);
    }
    constexpr uint8_t remainder = ((Nbit - 1) % radixbit) + 1;
    constexpr uint32_t size = 1U << remainder;
    constexpr uint32_t num_block = 1U << (Nbit - remainder);
    for (uint32_t block = 0; block < num_block; block++)
        INTTradixButterfly<remainder,true>(&res[size * block], size);
}

template <typename T = uint32_t, uint Nbit, bool modswitch>
inline void TwistMulInvert(std::array<DoubleSWord, 1 << Nbit> &res,
                           const std::array<T, 1 << Nbit> &a,
                           const std::array<SWord, 1 << Nbit> &twist)
{
    constexpr uint N = 1 << Nbit;
    for (int i = 0; i < N; i++) {
        if constexpr (modswitch){
            res[i] =
                (((static_cast<DoubleWord>(a[i]) * K) << shiftamount) +
                 a[i] + (1ULL << (32 - 1))) >>
                32;
        }else{
            res[i] = a[i];
        }
        res[i] = MulREDC(res[i], twist[i]);
    }
}

template <typename T, uint32_t Nbit, bool modsiwtch = false>
void TwistINTT(std::array<DoubleSWord, 1 << Nbit> &res,
               const std::array<T, 1 << Nbit> &a,
               const std::array<std::array<SWord, 1 << Nbit>,2> &table,
               const std::array<SWord, 1 << Nbit> &twist)
{
    TwistMulInvert<T, Nbit, modsiwtch>(res, a, twist);
    INTT<Nbit, 2>(res, table);
}

template <uint8_t radixbit>
inline void NTTradixButterfly(DoubleSWord *const res, const uint32_t size)
{
    static_assert(radixbit <= 1, "radix 2 is the maximum!");
    if constexpr (radixbit != 0) {
        // NTTradixButterfly<radixbit - 1>(&res[size / 2], size / 2);
        // NTTradixButterfly<radixbit - 1>(&res[0], size / 2);
        // const uint32_t block = size >> radixbit;
        // if constexpr (radixbit != 1)
        //     for (int i = 1; i < (1 << (radixbit - 1)); i++)
        //         for (int j = 0; j < block; j++)
        //             res[i * block + j + size / 2] =
        //                 res[i * block + j + size / 2]
        //                 << (3 * (64 - (i << (6 - radixbit))));
        ButterflyAddBothMod(res, size);
    }
}

template <uint32_t Nbit, uint8_t radixbit>
inline void NTTradix(DoubleSWord *const res, const uint32_t size,
                     const uint32_t num_block,
                     const std::array<SWord, 1 << Nbit> &table)
{
    for (uint32_t i = 1; i < (1 << radixbit); i++)
        TwiddleMul<Nbit>(&res[i * (size >> radixbit)], size >> radixbit,
                         BitReverse<radixbit>(i) * num_block, table);
    NTTradixButterfly<radixbit>(res, size);
}

template <uint32_t Nbit, uint8_t radixbit>
void NTT(std::array<DoubleSWord, 1 << Nbit> &res,
         const std::array<SWord, 1 << Nbit> &table)
{
    constexpr uint8_t remainder = ((Nbit - 1) % radixbit) + 1;
    constexpr uint size = 1U << remainder;
    constexpr uint num_block = 1U << (Nbit - remainder);
    for (uint block = 0; block < num_block; block++)
        NTTradixButterfly<remainder>(&res[size * block], size);
    for (uint8_t sizebit = remainder + radixbit; sizebit <= Nbit;
         sizebit += radixbit) {
        const uint size = 1U << sizebit;
        const uint num_block = 1U << (Nbit - sizebit);
        for (uint block = 0; block < num_block; block++)
            NTTradix<Nbit, radixbit>(&res[size * block], size, num_block,
                                     table);
    }
}

template <typename T = uint32_t, uint Nbit, bool modswitch>
inline void TwistMulDirect(std::array<T, 1 << Nbit> &res,
                           const std::array<DoubleSWord, 1 << Nbit> &a,
                           const std::array<SWord, 1 << Nbit> &twist)
{
    constexpr uint32_t N = 1 << Nbit;
    for (int i = 0; i < N; i++) {
        const SWord mulres = MulSREDC(a[i], twist[i]);
        res[i] = (mulres < 0) ? mulres + P : mulres;
        if constexpr (modswitch)
            res[i] = (static_cast<DoubleWord>(res[i]) * ((1ULL << 61) / P) +
                      (1ULL << (29 - 1))) >>
                     29;
    }
}

template <typename T, uint32_t Nbit, bool modsiwtch = false>
void TwistNTT(std::array<T, 1 << Nbit> &res, std::array<DoubleSWord, 1 << Nbit> &a,
              const std::array<std::array<SWord, 1 << Nbit>,2> &table,
              const std::array<SWord, 1 << Nbit> &twist)
{
    NTT<Nbit, 1>(a, table[0]);
    TwistMulDirect<T, Nbit, modsiwtch>(res, a, twist);
}

template <typename T, uint32_t Nbit, bool modswitcha, bool modswitchb>
void PolyMullvl1(std::array<T, 1 << Nbit> &res, std::array<T, 1 << Nbit> &a,
                 std::array<T, 1 << Nbit> &b,
                 const std::array<std::array<std::array<SWord, 1 << Nbit>, 2>, 2> &table,
                 const std::array<std::array<SWord, 1 << Nbit>, 2> &twist)
{
    std::array<DoubleSWord, 1 << Nbit> ntta, nttb;
    TwistINTT<T, Nbit, modswitcha>(ntta, a, table[1], twist[1]);
    TwistINTT<T, Nbit, modswitchb>(nttb, b, table[1], twist[1]);
    for (int i = 0; i < (1U << Nbit); i++)
        ntta[i] = MulSREDC(MulSREDC(ntta[i], R2), nttb[i]);
    TwistNTT<T, Nbit, modswitcha | modswitchb>(res, ntta, table[0], twist[0]);
}

}  // namespace raintt