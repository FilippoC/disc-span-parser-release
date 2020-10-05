#pragma once

inline float log_sum_exp(const float* const begin, const float* const end, const bool add_one = false)
{
    auto max_elem = *std::max_element(begin, end);
    auto sum = std::accumulate(begin, end, 0., [max_elem](float a, float b) { return a + std::exp(b - max_elem); });
    if (add_one)
        //return max_elem + log(sum + 1.);
        return max_elem + log1p(sum);
    else
        return max_elem + log(sum);
}

LogSemiring::LogSemiring() :
        value(0.f)
{}

LogSemiring::LogSemiring(const float v) :
        value(v)
{}

LogSemiring LogSemiring::null()
{
    return LogSemiring(-std::numeric_limits<float>::infinity());
}

LogSemiring LogSemiring::one()
{
    return LogSemiring(0.);
}

LogSemiring LogSemiring::from_exp(float x)
{
    return LogSemiring(log(x));
}

LogSemiring LogSemiring::from_log(float x)
{
    return LogSemiring(x);
}

LogSemiring LogSemiring::operator*(const LogSemiring& o) const
{
    return LogSemiring(value + o.value);
}

LogSemiring LogSemiring::operator+(const LogSemiring& o) const
{
    if (value > o.value)
        //return LogSemiring(value + log(1 + exp(o.value - value)));
        return LogSemiring(value + log1p(exp(o.value - value)));
    else
        //return LogSemiring(o.value + log(1 + exp(value - o.value)));
        return LogSemiring(o.value + log1p(exp(value - o.value)));
}

void LogSemiring::operator+=(const LogSemiring& o)
{
    if (value > o.value)
        //value += log(1. + exp(o.value - value));
        value += log1p(exp(o.value - value));
    else
        //value = (o.value + log(1 + exp(value - o.value)));
        value = (o.value + log1p(exp(value - o.value)));
}