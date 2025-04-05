#ifndef ERATOSTHENES_SIEVE_H
#define ERATOSTHENES_SIEVE_H

#ifdef __cplusplus
#include <assert.h>
#include <iostream>
#include <vector>
#endif

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

#endif
