#ifndef JUMP_SEARCH_H
#define JUMP_SEARCH_H

#ifdef __cplusplus
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#endif

/**
 * @brief jump search function
 * @param arr input array
 * @param key the element we want to search
 * @return int64_t index of key in the array if it exists, otherwise returns -1
 */
template <typename T> int64_t jump_search(std::vector<T> arr, T key) {
    int64_t n = arr.size(), step = sqrt(arr.size()), prev = 0;
    while (arr[std::min(step, n) - 1] < key) {
        prev = step;
        step += sqrt(n);
        if (prev >= n) {
            return -1; // out of bounds
        }
    }
    while (arr[prev] < key) {
        prev++;
        if (prev == std::min(step, n)) {
            return -1;
        }
    }
    return (arr[prev] == key) ? prev : -1;
}

#endif
