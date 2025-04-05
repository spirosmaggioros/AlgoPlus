#ifndef PARTITIONS_H
#define PARTITIONS_H

#ifdef __cplusplus
#include <iostream>
#include <vector>
#endif

void iterate(long long n, std::vector<std::vector<long long>>& ans,
             std::vector<long long> prefix = {}) {
    if (n == 0) {
        ans.push_back(prefix);
    } else {
        long long max = prefix.size() ? std::min(prefix.back(), n) : n;
        prefix.push_back(1);

        for (long long i = 1; i <= max; i++) {
            prefix.back() = i;
            iterate(n - i, ans, prefix);
        }
    }
}

std::vector<std::vector<long long>> partitions(long long n) {
    std::vector<std::vector<long long>> ans;
    iterate(n, ans);
    return ans;
}

#endif
