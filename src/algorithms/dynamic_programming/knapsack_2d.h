#ifndef KNAPSACK_H
#define KNAPSACK_H

#ifdef __cplusplus
#include <iostream>
#include <utility>
#include <vector>
#endif

/**
 * @brief knapsack 0/1 function
 * @param v the passed vector
 * @param capacity the capacity of the knapsack
 * @return int: the maximum value you can collect with a knapsack of the passed
 * capacity
 */
int knapsack(std::vector<std::pair<int, int>>& v, int capacity) {
    if (capacity == 0) {
        return 0;
    }

    int n = int(v.size());
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1));

    for (int i = 1; i <= n; i++) {
        for (int j = 0; j <= capacity; j++) {
            dp[i][j] = dp[i - 1][j];
            if (i == 0 || j == 0) {
                dp[i][j] = 0;
            } else if (v[i - 1].first <= j) {
                dp[i][j] = std::max(dp[i - 1][j], v[i - 1].second + dp[i - 1][j - v[i - 1].first]);
            }
        }
    }

    return dp[n][capacity];
}

#endif
