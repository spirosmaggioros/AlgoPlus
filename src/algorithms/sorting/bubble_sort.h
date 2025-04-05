#ifndef BUBBLE_SORT_H
#define BUBBLE_SORT_H

#ifdef __cplusplus
#include <cstdint>
#include <iostream>
#include <vector>
#endif

/**
 * @brief bubble sort function
 * @param arr input array
 */
template <typename T> void bubble_sort(std::vector<T>& arr) {
    for (int64_t i = 0; i < arr.size(); i++) {
        bool check = 0;
        for (int64_t j = 0; j < arr.size() - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
                check = 1;
            }
        }
        if (!check) {
            break;
        }
    }
}

#endif
