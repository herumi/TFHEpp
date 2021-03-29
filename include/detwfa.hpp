#pragma once

#include <trgsw.hpp>
#include <utils.hpp>

namespace TFHEpp {
template <class P>
inline void CMUXFFT(TRLWE<P> &res, const TRGSWFFT<P> &cs, const TRLWE<P> &c1,
                    const TRLWE<P> &c0)
{
    for (int i = 0; i < P::n; i++) {
        res[0][i] = c1[0][i] - c0[0][i];
        res[1][i] = c1[1][i] - c0[1][i];
    }
    trgswfftExternalProduct<P>(res, res, cs);
    for (int i = 0; i < P::n; i++) {
        res[0][i] += c0[0][i];
        res[1][i] += c0[1][i];
    }
}

template <class P>
inline void CMUXFFTwithPolynomialMulByXaiMinusOne(TRLWE<P> &acc,
                                                  const TRGSWFFT<P> &cs,
                                                  const typename P::T a)
{
    TRLWE<P> temp;
    PolynomialMulByXaiMinusOne<P>(temp[0], acc[0], a);
    PolynomialMulByXaiMinusOne<P>(temp[1], acc[1], a);
    trgswfftExternalProduct<P>(temp, temp, cs);
    for (int i = 0; i < P::n; i++) {
        acc[0][i] += temp[0][i];
        acc[1][i] += temp[1][i];
    }
}

}  // namespace TFHEpp