#ifndef KADANE_H
#define KADANE_H

#ifdef __cplusplus
#include <climits>
#include <cstdint>
#include <iostream>
#include <vector>
#endif

/**
 * @brief kadane's algorithm
 * @details Computes the maximum continuous subarray sum of an array
 * @param arr input array
 * @return int64_t max contiguous sum of the array
 */
int64_t kadane(std::vector<int> arr) {
    int64_t mmax = INT_MIN, temp = 0;
    for (int64_t i = 0; i < arr.size(); i++) {
        temp += arr[i];
        if (temp > mmax) {
            mmax = temp;
        }
        if (temp < 0) {
            temp = 0;
        }
    }
    return mmax;
}

#endif
