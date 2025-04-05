#ifndef SUBSET_SUM_H
#define SUBSET_SUM_H

#ifdef __cplusplus
#include <cassert>
#include <iostream>
#include <vector>
#endif

/**
 * @brief subset sum function
 * @details Returns the number of subsets you can create with sum of B
 * @param v: the passed vector
 * @param B: the sum
 * @returns: int, the total number of subsets you can create
 */
int subset_sum(std::vector<int>& v, int B) {
    if (int(v.size()) == 0) {
        return 0;
    }
    int n = int(v.size());
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(B + 1, 0));
    for (int i = 0; i <= n; i++) {
        dp[i][0] = 1;
    }

    for (int i = 1; i <= n; i++) {
        for (int j = 0; j <= B; j++) {
            dp[i][j] = dp[i - 1][j];

            if (j - v[i - 1] >= 0) {
                dp[i][j] += dp[i - 1][j - v[i - 1]];
            }
        }
    }

    return dp[n][B];
}

#endif
