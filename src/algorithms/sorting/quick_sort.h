#ifndef ALGOPLUS_QUICK_SORT_H
#define ALGOPLUS_QUICK_SORT_H

#include <algorithm> // For std::partition and std::swap
#include <iterator>  // For std::iterator_traits

/**
* @brief Finds and returns the median of three elements of a container.
* @param a An iterator pointing to the first element.
* @param b An iterator pointing to the second element.
* @param c An iterator pointing to the third element.
* @details This function takes iterators to three elements in a container,
* compares these elements, and returns an iterator pointing to the median
* value. The function doesn't mutate the original elements.
* @return An iterator pointing to the median of the three input element.
*/
template <typename Iter> Iter median_of_three(Iter a, Iter b, Iter c) {
    if (*a < *b) {
        if (*b < *c)
        return b;
        else if (*a < *c)
        return c;
        else
        return a;
    } else {
        if (*a < *c)
        return a;
        else if (*b < *c)
        return c;
        else
        return b;
    }
}

/**
* @brief Sorts a range of elements using the Quick Sort algorithm
* @param begin An iterator to the beginning of the range to be sorted.
* @param end An iterator to the end of the range to be sorted.
* @details The implementation requires that the iterator type `Iter` supports
* Random Access to enable efficient partitioning and element swapping. The
* elements in the range must be move-assignable and should support comparison
* with the '<' operator. If these requirements are not met, the behavior of the
* function is undefined. This function sorts the elements in the range [begin,
* end) into ascending order. The sort is performed in-place. Its performance is
* generally O(n log n) on average, though worst-case performance is O(n^2)
*/
template <typename Iter> void quick_sort(Iter begin, Iter end) {
    auto distance = std::distance(begin, end);
    if (distance <= 1) {
        return;
    }

    // choose the pivot as median of first, middle and last element
    Iter mid = begin + distance / 2;
    Iter pivot_pos = median_of_three(begin, mid, end - 1);
    auto pivot = *pivot_pos;

    // swap pivot to start
    std::swap(*begin, *pivot_pos);

    Iter middle1 = std::partition(
        begin + 1, end, [pivot](const auto &elem) { return elem < pivot; });

    Iter middle2 = std::partition(
        middle1, end, [pivot](const auto &elem) { return !(pivot < elem); });

    // swap pivot to its final place
    std::swap(*begin, *(middle1 - 1));

    // recursively apply quick_sort to partitions
    quick_sort(begin, middle1 - 1);
    quick_sort(middle2, end);
}

#endif // QUICK_SORT_H
