#pragma once

#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>

// add_one is usefull to take into account empty spans
// as of now this function is not used
inline float log_sum_exp(const float* const begin, const float* const end, const bool add_one);

struct LogSemiring
{
    float value;

    inline LogSemiring();

    inline static LogSemiring null();
    inline static LogSemiring one();

    inline static LogSemiring from_exp(float x);
    inline static LogSemiring from_log(float x);

    inline LogSemiring operator*(const LogSemiring& o) const;
    inline LogSemiring operator+(const LogSemiring& o) const;
    inline void operator+=(const LogSemiring& o);

protected:
    explicit inline LogSemiring(const float v);
};

#include "log-impl.h"