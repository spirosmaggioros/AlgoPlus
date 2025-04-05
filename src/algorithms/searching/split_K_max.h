#ifndef SPLIT_K_MAX_H
#define SPLIT_K_MAX_H

#ifdef __cplusplus
#include <algorithm>
#include <climits>
#include <iostream>
#include <numeric>
#include <vector>
#endif

namespace helpers {
/**
 * @brief check function(helper) for minimum_max_sub_sum
 * @param v: vector<int> the passed array
 * @param mid: the mid of each binary search step of minimum_max_sub_sum
 * @param K: the total subarrays
 * @returns true: if the total count of subarrays is <= K(if there's a way to do
 * better)
 * @return false: if there's no splits better
 */
bool check(std::vector<int>& v, int mid, int K) {
    int sum = 0;
    int splits = 1;

    for (int i = 0; i < int(v.size()); i++) {
        if (v[i] > mid) {
            return false;
        }

        sum += v[i];
        if (sum > mid) {
            splits++;
            sum = v[i];
        }
    }

    return (splits <= K) ? true : false;
}
} // namespace helpers

/**
 * @brief minimum max sub sum function. Finds the maximum sum of each K
 * subarrays and returns the minimum of them. Basically, answers the question:
 * "What is the best way of splitting a field to your K sons - without changing
 * the order - so that you are as fair as possible"
 *
 * @param v: vector<int> the passed array
 * @param K: the total subarrays
 * @returns int: the min(max(subarrays sum))
 */
int minimum_max_sub_sum(std::vector<int>& v, int K) {
    int start = v[0];
    int end = std::accumulate(v.begin(), v.end(), 0);

    int ans = INT_MAX;
    while (start <= end) {
        int mid = start + (end - start) / 2;

        if (helpers::check(v, mid, K)) {
            ans = mid;
            end = mid - 1;
        } else {
            start = mid + 1;
        }
    }

    return ans;
}

#endif
