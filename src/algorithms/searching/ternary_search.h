#ifndef TERNARY_SEARCH_H
#define TERNARY_SEARCH_H

#ifdef __cplusplus
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#endif

namespace {
/**
 * @brief Performs a ternary search between two iterators.
 *
 * @details The function uses the ternary search algorithm to find a specified
 * value within the range [first, last). The ternary search splits the search
 * space into three parts and eliminates the non-possible regions in each
 * iteration.
 *
 * @tparam Iter type of the iterator
 * @tparam T type of the value to be searched
 * @param first Iterator pointing to start of the range in which to search for
 * the value.
 * @param last Iterator pointing to end of the range in which to search for the
 * value.
 * @param value The value to search for in the range.
 * @return Iterator pointing to the found value, or `last` if the value is not
 * found.
 */
template <typename Iter, typename T> Iter ternary_search(Iter first, Iter last, T value) {
    while (first < last) {
        typedef typename std::iterator_traits<Iter>::difference_type diff_t;
        diff_t diff = std::distance(first, last) / 3;
        Iter mid1 = first + diff;
        Iter mid2 = mid1 + diff;

        if (*mid1 == value)
            return mid1; // Found value
        else if (*mid2 == value)
            return mid2; // Found value

        if (value < *mid1)
            last = mid1;
        else if (value > *mid2)
            first = mid2 + 1;
        else {
            first = mid1 + 1;
            last = mid2;
        }
    }
    return last; // Not found
}
} // namespace
/**
 * @brief Performs an interpolation search on a sorted container.
 *
 * @details
 * This function performs a ternary_search algorithm,
 * which is a divide-and-conquer search algorithm and works by
 * dividing the search space into three equal parts and determining
 * in which part the key lies. This algorithm as a time complexity of O(log3 N)
 *
 * @tparam Container A type of container.
 * @tparam T The type of elements in the sequence to be searched.
 * @param c The container to be searched in.
 * @param key The key value to search for in the sequence.
 * @return If the key is found, returns the index. If the key is not found,
 * returns -1.
 */
template <typename Container, typename T> int64_t ternary_search(Container& c, const T& value) {
    auto it = ternary_search(c.begin(), c.end(), value);
    if (it != c.end() && *it == value)
        return static_cast<int64_t>(std::distance(c.begin(), it));
    else
        return -1;
}

#endif // TERNARY_SEARCH_H
