#ifndef LINEAR_SEARCH_H
#define LINEAR_SEARCH_H

#ifdef __cplusplus
#include <iostream>
#include <vector>
#endif

/**
 * @brief linear search function
 * @param arr input array
 * @param key the element we want to search
 * @return true if key exists in the array
 * @return false otherwise
 */
template <typename T> bool linear_search(std::vector<T> arr, T key) {
    return std::find_if(arr.begin(), arr.end(), [key](const auto& x) { return x == key; }) !=
           arr.end();
}

#endif
