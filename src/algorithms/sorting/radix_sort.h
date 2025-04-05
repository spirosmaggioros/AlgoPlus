#ifndef RADIX_SORT_H
#define RADIX_SORT_H

#ifdef __cplusplus
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#endif

namespace {
/**
 * @brief Perform counting sort for a given vector of elements using a specific
 * radix.
 *
 * @details
 * This function implements the counting sort algorithm for a given vector of
 * elements using a specific radix. It sorts the elements based on their digits
 * at the specified radix position.
 *
 * @tparam T The data type of the elements in the vector.
 * @param arr The vector to be sorted.
 * @param exp The radix position to consider (e.g., 1 for units digit, 10 for
 * tens digit, etc.).
 */
template <typename T> void countingSortForRadix(std::vector<T>& arr, int exp) {
    int64_t n = arr.size();
    std::vector<T> output(n);
    std::vector<int> count(10, 0);

    for (int64_t i = 0; i < n; i++) {
        count[(arr[i] / exp) % 10]++;
    }

    for (int64_t i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }

    for (int64_t i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

    arr = output;
}
} // namespace

/**
 * @brief Perform radix sort on a given vector of elements.
 *
 * @details
 * This function implements the radix sort algorithm on a given vector of
 * elements. It sorts the elements based on their digits from least significant
 * to most significant.
 *
 * @tparam T The data type of the elements in the vector.
 * @param arr The vector to be sorted.
 */
template <typename T> void radix_sort(std::vector<T>& arr) {
    T maxElement = *max_element(arr.begin(), arr.end());

    for (int exp = 1; maxElement / exp > 0; exp *= 10) {
        countingSortForRadix(arr, exp);
    }
}

#endif // RADIX_SORT_H
