#ifndef SEGMENT_TREE_H
#define SEGMENT_TREE_H

#ifdef __cplusplus
#include <cmath>
#include <iostream>
#include <vector>
#endif

/**
 * @brief segment tree class
 * Usefull for fast query sums and updates
 */
class segment_tree {
  public:
    /**
     * @brief default constructor for segment tree
     * @param v: input vector
     */
    inline explicit segment_tree(const std::vector<int>& v) noexcept : data(v) {
        int x = (int)(ceil(log2(v.size())));
        int max_size = 2 * (int)(pow(2, x) - 1);
        this->root = std::vector<int>(max_size);
        _construct(0, v.size() - 1, 0);
    }

    /**
     * @update function
     * @param idx: the index we want to update
     * @param key: the new key(i.e. v[idx] = key)
     */
    inline void update(int idx, int key) {
        if (idx < 0 || idx > data.size()) {
            return;
        }

        int diff = key - data[idx];
        data[idx] = key;
        _update(0, data.size() - 1, idx, diff, 0);
    }

    /**
     * @brief sum function
     * @param query_start: the starting index
     * @param query_end: the ending index
     * @return int: the sum of the window [query_start, query_end]
     */
    inline int sum(int query_start, int query_end) {
        if (query_start < 0 || query_end >= data.size() || query_start > query_end) {
            return -1;
        }

        return _sum(0, data.size() - 1, query_start, query_end, 0);
    }

  private:
    std::vector<int> root;
    std::vector<int> data;

    /**
     * @brief construct() helper function
     * @param start: the starting index
     * @param end: the ending index
     * @param si: the current index
     * @return int: the value of the vector at index si(i.e v[si])
     */
    int _construct(int start, int end, int si = 0) {
        if (start == end) {
            root[si] = data[start];
            return data[start];
        }

        int mid = start + (end - start) / 2;
        root[si] = _construct(start, mid, si * 2 + 1) + _construct(mid + 1, end, si * 2 + 2);
        return root[si];
    }

    /**
     * @brief update() helper function
     * @param start: the starting index
     * @param end: the ending index
     * @param diff: the difference of the new value from the vector's value
     * @param si: the current index
     */
    void _update(int start, int end, int i, int diff, int si = 0) {
        if (i < start || i > end) {
            return;
        }

        root[si] = root[si] + diff;
        if (end != start) {
            int mid = start + (end - start) / 2;
            _update(start, mid, i, diff, si * 2 + 1);
            _update(mid + 1, end, i, diff, si * 2 + 2);
        }
    }

    /**
     * @brief sum() helper function
     * @param start: the starting index
     * @param end: the ending index
     * @param query_start: the query starting index
     * @param query_end: the query ending index
     * @param si: the current index
     * @return int: the total sum of the window
     */
    int _sum(int start, int end, int query_start, int query_end, int si = 0) {
        if (query_start <= start && query_end >= end) {
            return root[si];
        }
        if (end < query_start || start > query_end) {
            return 0;
        }
        int mid = start + (end - start) / 2;
        return _sum(start, mid, query_start, query_end, si * 2 + 1) +
               _sum(mid + 1, end, query_start, query_end, si * 2 + 2);
    }
};

#endif
