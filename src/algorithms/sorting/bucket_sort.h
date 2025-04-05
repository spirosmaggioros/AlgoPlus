#ifndef BUCKET_SORT_H
#define BUCKET_SORT_H

#ifdef __cplusplus
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#endif

/**
 * @brief bucket sort function
 * @param arr input array
 */
template <typename T> void bucket_sort(std::vector<T>& arr) {
    if (std::is_same_v<T, int> || std::is_same_v<T, int64_t>) {
        std::sort(arr.begin(), arr.end());
        return;
    }
    std::vector<float> buckets[arr.size()];
    for (int64_t i = 0; i < arr.size(); i++) {
        int64_t index = arr.size() * arr[i];
        buckets[index].push_back(arr[i]);
    }

    for (int64_t i = 0; i < arr.size(); i++) {
        std::sort(buckets[i].begin(), buckets[i].end());
    }

    int64_t index = 0;
    for (int64_t i = 0; i < arr.size(); i++) {
        for (int64_t j = 0; j < buckets[i].size(); j++) {
            arr[index++] = buckets[i][j];
        }
    }
}

#endif
