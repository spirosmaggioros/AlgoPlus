#ifndef HEAP_SORT_H
#define HEAP_SORT_H

#ifdef __cplusplus
#include <cstdint>
#include <iostream>
#include <vector>
#endif

/**
 * @brief heapify method
 * @param arr input array
 * @param parent parent of the index
 */
template <typename T> static void heapify_method(std::vector<T>& arr, int64_t n, int64_t parent) {

    // largest node
    int64_t largest = parent;

    // left child
    int64_t left = 2 * parent + 1;

    // right child
    int64_t right = 2 * parent + 2;

    // Compare left child with parent
    if (left < n && arr[left] > arr[largest]) {
        largest = left;
    }
    if (right < n && arr[right] > arr[largest]) {
        largest = right;
    }
    if (largest != parent) {
        std::swap(arr[parent], arr[largest]);

        heapify_method(arr, n, largest);
    }
}

/**
 * @brief heap sort function
 * @param arr input array
 */
template <typename T> void heap_sort(std::vector<T>& arr) {
    int64_t n = arr.size();
    // heapify the array at first
    for (int64_t i = n / 2 - 1; i >= 0; i--) {
        heapify_method(arr, n, i);
    }
    for (int64_t i = n - 1; i >= 0; i--) {
        std::swap(arr[0], arr[i]);

        heapify_method(arr, i, 0);
    }
}

#endif
