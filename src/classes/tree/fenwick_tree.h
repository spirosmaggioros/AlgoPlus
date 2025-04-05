#ifndef FENWICK_TREE_H
#define FENWICK_TREE_H

#ifdef __cplusplus
#include <iostream>
#include <vector>
#endif

/**
 * @brief fenwick tree class
 */
template <typename T> struct fenwick_tree {
    std::vector<T> tree;
    int n;

    /**
     * @brief default constructor of fenwick tree class
     * @param v: the input vector
     */
    inline explicit fenwick_tree(const std::vector<T>& v) noexcept : n(int(v.size())) {
        tree = std::vector<T>(n, 0);
        for (int i = 0; i < n; i++) {
            this->update(i, v[i]);
        }
    }

    /**
     * @brief sum query function
     * @param k: the ending index of the query
     * @return T: the sum of range [0, k]
     */
    inline T sum(int k) {
        T sum = 0;
        for (; k >= 0; k = (k & (k + 1)) - 1) {
            sum += tree[k];
        }
        return sum;
    }

    /**
     * @brief sum query function(from index a to b)
     * @param a: starting index
     * @param b: ending index
     * @returns T: the sum of range [a, b]
     */
    inline T sum(int a, int b) { return sum(b) - sum(a - 1); }

    /**
     * @brief update query function
     * @param k: the index
     * @param x: the value that will be added to data[k]
     */
    inline void update(int k, int x) {
        for (; k < n; k = k | (k + 1)) {
            tree[k] += x;
        }
    }
};

#endif
