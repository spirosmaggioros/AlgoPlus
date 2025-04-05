#ifndef COIN_CHANGE_H
#define COIN_CHANGE_H

#ifdef __cplusplus
#include <cstdint>
#include <iostream>
#include <vector>
#endif

/**
 * @brief minimum cost function for the coin change problem
 *
 * @param arr input array
 * @return int64_t the minimum cost
 */
int64_t min_cost(std::vector<int> arr, int N) {
    int max_of = N + 1;
    std::vector<int> dp(N + 1, max_of);
    dp[0] = 0;
    for (int i = 0; i <= N; i++) {
        for (auto& x : arr) {
            if (x <= i) {
                dp[i] = std::min(dp[i], 1 + dp[i - x]);
            }
        }
    }
    return dp[N] > N ? -1 : dp[N];
}

#endif
