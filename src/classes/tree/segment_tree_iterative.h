#ifndef SEGMENT_TREE_ITERATIVE_H
#define SEGMENT_TREE_ITERATIVE_H

#ifdef __cplusplus
#include <cassert>
#include <iostream>
#include <vector>
#endif

/**
 * @brief segment tree(iterative version)
 */
template <typename T> struct seg_tree {
    std::vector<T> tree;
    std::vector<T> data;
    int n;

    /**
     * @brief default constructor
     * @param v: the input vector
     */
    inline explicit seg_tree(const std::vector<T>& v) noexcept : data(v), n(int(v.size())) {
        tree = std::vector<T>(2 * v.size(), 0);
        int idx = n - 1;
        for (int i = 2 * n - 1; i >= n; i--) {
            tree[i] = data[idx--];
        }

        idx = 0;
        for (int i = n - 1; i >= 1; i--) {
            tree[i] = tree[2 * n - idx - 1] + tree[2 * n - idx - 2];
            idx += 2;
        }
    }

    /**
     * @brief sum query
     * @param a: the first index
     * @param b: the second index
     * @details: indexes are allowed to be from [0, n)
     * @return: the sum of the range[a, b]
     */
    inline T sum(int a, int b) {
        assert(a >= 0 && a <= b && b < n);
        a += n;
        b += n;
        T s = 0;
        while (a <= b) {
            if (a % 2 == 1) {
                s += tree[a++];
            }
            if (b % 2 == 0) {
                s += tree[b--];
            }
            a /= 2;
            b /= 2;
        }

        return s;
    }

    /**
     * @brief update query
     * @param idx: the passed index
     * @param x: the new value of the index
     */
    inline void update(int idx, T x) {
        assert(idx >= 0 && idx < n);
        idx += n;
        tree[idx] = x;
        for (idx /= 2; idx >= 1; idx /= 2) {
            tree[idx] = tree[2 * idx] + tree[2 * idx + 1];
        }
    }
};

#endif
