#ifndef MIN_DISTANCE_H
#define MIN_DISTANCE_H

#ifdef __cplusplus
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#endif

/**
 * @brief edit distance function
 * @param word1 first string
 * @param word2 second string
 * @return int64_t the minimum steps to make word1 equal to word2(i.e. the
 * distance of word1 and word2)
 */
int64_t min_dist(const std::string word1, const std::string word2) {
    if (word1.size() == 0 && word2.size() == 0) {
        return 0;
    }
    if (word1.size() == 0 && word2.size() != 0) {
        return word2.size();
    }
    if (word1.size() != 0 && word2.size() == 0) {
        return word1.size();
    }
    int n = word1.size(), w = word2.size();
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(w + 1));
    for (int i = 0; i <= n; i++) {
        dp[i][0] = i;
    }

    for (int i = 0; i <= w; i++) {
        dp[0][i] = i;
    }
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= w; j++) {
            if (word1[i - 1] == word2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = std::min(dp[i - 1][j - 1], std::min(dp[i - 1][j], dp[i][j - 1])) + 1;
            }
        }
    }

    return dp[n][w];
}

#endif
