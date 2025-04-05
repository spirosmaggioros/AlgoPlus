#ifndef SELECTION_SORT_H
#define SELECTION_SORT_H

#ifdef __cplusplus
#include <cstdint>
#include <iostream>
#include <vector>
#endif

/**
 * @brief selection sort function
 * @param arr input array
 */
template <typename T> void selection_sort(std::vector<T>& arr) {
    int64_t min_idx = 0;
    for (int64_t i = 0; i < arr.size() - 1; i++) {
        min_idx = i;
        for (int64_t j = i + 1; j < arr.size(); j++) {
            if (arr[j] < arr[min_idx]) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            std::swap(arr[min_idx], arr[i]);
        }
    }
}

#endif
