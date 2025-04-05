#ifndef ALGOPLUS_COUNTING_SORT_H
#define ALGOPLUS_COUNTING_SORT_H

#ifdef __cplusplus
#include <algorithm>
#include <cstdint>
#include <vector>
#endif

/**
 * @brief Performs counting sort on a vector of elements.
 * @tparam T The type of elements in the vector.
 * @param arr The vector to be sorted.
 */
template <typename T> void counting_sort(std::vector<T>& arr) {
    T maxElement = std::max(0, *max_element(arr.begin(), arr.end()));
    std::vector<T> count(maxElement + 1, 0);
    std::vector<T> output(arr.size());

    // Store count of each element
    for (int64_t i = 0; i < arr.size(); i++) {
        count[arr[i]]++;
    }

    // Change count[i] so that count[i] now contains actual position of this
    // element in output array
    for (int64_t i = 1; i <= maxElement; i++) {
        count[i] += count[i - 1];
    }

    // Build the output array
    for (int64_t i = arr.size() - 1; i >= 0; i--) {
        output[count[arr[i]] - 1] = arr[i];
        count[arr[i]]--;
    }

    arr = output;
}

#endif // ALGOPLUS_COUNTING_SORT_H
