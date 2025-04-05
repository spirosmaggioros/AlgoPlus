#ifndef RMQ_SEGMENT_TREE_H
#define RMQ_SEGMENT_TREE_H

#ifdef __cplusplus
#include <cassert>
#include <iostream>
#include <vector>
#endif

#ifndef count_leading_zeroes
#if defined(_MSC_VER)
#if defined(_M_ARM) || defined(_M_ARM64)
#define count_leading_zeroes(x) _CountLeadingZeros(x);
#else
#define count_leading_zeroes(x) __lzcnt(x);
#endif
#else
#define count_leading_zeroes(x) __builtin_clz(x);
#endif
#endif

/**
 * @brief credits to @nealwu for his RMQ query
 * RMQ struct for range query minimum
 */
template <typename T, bool maximum_mode = false> struct RMQ {
    static int highest_bit(unsigned x) { return x == 0 ? -1 : 31 - count_leading_zeroes(x); }

    int n = 0;
    std::vector<T> values;
    std::vector<std::vector<int>> range_low;

    RMQ(const std::vector<T>& _values = {}) {
        if (!_values.empty())
            build(_values);
    }

    // Note: when `values[a] == values[b]`, returns b.
    // Need to change this if you want to return a instead of b
    int better_index(int a, int b) const {
        return (maximum_mode ? values[b] < values[a] : values[a] < values[b]) ? a : b;
    }

    void build(const std::vector<T>& _values) {
        values = _values;
        n = int(values.size());
        int levels = highest_bit(n) + 1;
        range_low.resize(levels);

        for (int k = 0; k < levels; k++)
            range_low[k].resize(n - (1 << k) + 1);

        for (int i = 0; i < n; i++)
            range_low[0][i] = i;

        for (int k = 1; k < levels; k++)
            for (int i = 0; i <= n - (1 << k); i++)
                range_low[k][i] =
                    better_index(range_low[k - 1][i], range_low[k - 1][i + (1 << (k - 1))]);
    }

    // Note: breaks ties by choosing the largest index.
    int query_index(int a, int b) const {
        assert(0 <= a && a < b && b <= n);
        int level = highest_bit(b - a);
        return better_index(range_low[level][a], range_low[level][b - (1 << level)]);
    }

    T query_value(int a, int b) const { return values[query_index(a, b)]; }
};

#endif
