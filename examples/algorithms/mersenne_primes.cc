#ifdef __cplusplus
#include "../../src/algorithms/number_theory/mersenne_primes.h"
#endif

int main() {
    int64_t n = 100;
    std::vector<int> first_10000000_mersenne_primes = mersenne(10000000);
    for (auto& x : first_10000000_mersenne_primes) {
        std::cout << x << " ";
    }
}