#ifndef FIB_H
#define FIB_H

#ifdef __cplusplus
#include <cstdint>
#include <iostream>
#include <math.h>
#include <vector>
#endif

/**
 * @brief fibonacci function
 *
 * @param n upper boumd
 * @return int64_t returns the total fibonacci numbers till n
 */
int64_t fibonacci(int64_t n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

/**
 * @brief fibonacci with dp function
 *
 * @param n upper boumd
 * @return int64_t returns the total fibonacci numbers till n
 */
int64_t fibonacci_dp(int64_t n) {
    std::vector<int64_t> dp(n + 2);
    dp[0] = 0;
    dp[1] = 1;
    for (int64_t i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}

/**
 * @brief fibonacci with bottom up dp function
 *
 * @param n upper boumd
 * @return int64_t returns the total fibonacci numbers till n
 */
int64_t fibonacci_bottom_up(int64_t n) {
    int64_t a = 0, b = 1, c = 0;
    if (n == 0) {
        return 0;
    }
    for (int64_t i = 2; i <= n; i++) {
        c = a + b;
        a = b;
        b = c;
    }
    return b;
}

/**
 * @brief fibonacci using binet's formula(golden ratio) function
 *
 * @param n upper boumd
 * @return int64_t returns the total fibonacci numbers till n
 */
int64_t fibonacci_binet(int64_t n) {
    double phi = (std::sqrt(5) + 1) / 2;
    return (int64_t)std::round(std::pow(phi, n) / sqrt(5));
}
#endif
