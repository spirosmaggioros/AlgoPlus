#ifndef INSERTION_SORT_H
#define INSERTION_SORT_H

#ifdef __cplusplus
#include <cstdint>
#include <iostream>
#include <vector>
#endif

/**
 * @brief insertion sort function
 * @param arr input array
 */
template <typename T> void insertion_sort(std::vector<T>& arr) {
    for (int64_t i = 1; i < arr.size(); i++) {
        int64_t j = i - 1;
        for (; j >= 0 && arr[j] > arr[i]; j--) {
            arr[j + 1] = arr[j];
        }
        arr[j + 1] = arr[i];
    }
}

#endif
