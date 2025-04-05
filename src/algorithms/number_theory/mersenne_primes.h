#ifndef MERSENNE_PRIMES_H
#define MERSENNE_PRIMES_H

#ifdef __cplusplus
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#endif

namespace helpers {
/**
 * @brief eratosthenes sieve function(sieve of eratosthenes)
 *
 * @param n upper bound
 * @return std::vector<bool> if arr[i] = 1 then i is prime
 */
std::vector<bool> soe(int64_t n) {
    assert(n != 0);
    std::vector<bool> prime(n + 1, true);
    for (int64_t p = 2; p * p <= n; ++p) {
        if (prime[p]) {
            for (int64_t i = p * p; i <= n; i += p) {
                prime[i] = false;
            }
        }
    }
    return prime;
}
} // namespace helpers

/**
 * @brief mersenne prime function
 *
 * @param n the upper bound
 * @return std::vector<int> the total mersenne primes till the upper bound n
 */
std::vector<int64_t> mersenne(int64_t n) {
    std::vector<bool> prime = helpers::soe(n + 1);
    std::vector<int64_t> elements;

    for (int k = 2; ((1LL << k) - 1) <= n; k++) {
        int64_t val = (1LL << k) - 1;
        if (val <= n && prime[val]) {
            elements.push_back(k);
        }
    }

    return elements;
}

#endif
