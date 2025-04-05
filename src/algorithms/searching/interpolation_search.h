#ifndef INTERPOLATION_SEARCH_H
#define INTERPOLATION_SEARCH_H

#ifdef __cplusplus
#include <algorithm>
#include <cstdint>
#include <vector>
#endif

/**
 * @brief Performs an interpolation search on a sorted container.
 *
 * @details
 * Interpolation Search estimates the position of the key in the sorted
 * container and uses this estimation to dramatically reduce the search time for
 * certain data distributions. However, this function assumes that the elements
 * are uniformly distributed and works best when this assumption holds. On
 * average, its complexity is O(log(log(n))), but in the worst case, it can
 * become O(n).
 *
 * @tparam Iterator The type of iterator for the container.
 * @tparam T The type of the object (key) to be searched.
 * @param begin The iterator pointing to the beginning of the container.
 * @param end The iterator pointing to the end of the container.
 * @param key The key to be searched within the container.
 * @return Returns the index (location) of the key in the container if found, or
 * -1 if the key is not found.
 */
template <typename Iterator, typename T>
int64_t interpolation_search(Iterator begin, Iterator end, T key) {
    auto size = std::distance(begin, end);

    auto it_begin = begin;
    auto it_end = end - 1;

    while (size > 0 && *it_begin <= key && *it_end >= key) {
        auto guess = it_begin;
        if (*it_end != *it_begin) {
            auto offset = static_cast<int>(static_cast<double>(size - 1) * (key - *it_begin) /
                                           (*it_end - *it_begin));
            std::advance(guess, offset);
        }
        if (*guess == key) {
            return std::distance(begin, guess);
        }

        if (*guess > key) {
            it_end = guess - 1;
        } else {
            it_begin = guess + 1;
        }
        size = std::distance(it_begin, it_end) + 1;
    }

    return static_cast<int64_t>(-1);
}

#endif // INTERPOLATION_SEARCH_H
