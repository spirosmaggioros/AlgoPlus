#ifndef BINARY_SEARCH_H
#define BINARY_SEARCH_H

#ifdef __cplusplus
#include <cstdint>
#include <iostream>
#include <vector>
#endif

/**
 * @brief binary search function
 *
 * @param arr The vector
 * @param left The left pointer
 * @param right The right pointer
 * @param x The element we want to search
 * @return int64_t The index of the element in the vector, if it doesn't exist
 * then it returns -1
 */
template <typename T> int64_t bin_search(std::vector<T> arr, int64_t left, int64_t right, T x) {
    while (left <= right) {
        int64_t mid = left + (right - left) / 2;
        if (arr[mid] == x) {
            return mid;
        }
        if (arr[mid] < x) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1; // element not found
}

/**
 * @brief lower bound search function
 *
 * @param arr The vector
 * @param left The left pointer
 * @param right The right pointer
 * @param x The element we want to search
 * @return int64_t The index of the first element that is not less than x
 */
template <typename T> int64_t lower_bound(std::vector<T> arr, int64_t left, int64_t right, T x) {
    int64_t result = right, mid = 0;
    --right;

    while (left <= right) {
        mid = left + (right - left) / 2;
        if (arr[mid] >= x) {
            result = mid;
            // look for smaller index on the left
            right = mid - 1;
        } else {
            left = mid + 1; // look on the right
        }
    }
    return result;
}

/**
 * @brief upper bound search function
 *
 * @param arr The vector
 * @param left The left pointer
 * @param right The right pointer
 * @param x The element we want to search
 * @return int64_t The index of the first element that is greater than x
 */
template <typename T> int64_t upper_bound(std::vector<T> arr, int64_t left, int64_t right, T x) {
    int64_t result = right, mid = 0;
    --right;

    while (left <= right) {
        mid = left + (right - left) / 2;
        if (arr[mid] > x) {
            result = mid;
            // look for smaller index on the left
            right = mid - 1;
        } else {
            left = mid + 1; // look on the right
        }
    }
    return result;
}

#endif
