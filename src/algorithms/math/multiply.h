#pragma once

#ifdef __cplusplus
#include <iostream>
#include <vector>
#endif

/*
 * @brief mul: multiplies two 2D vectors
 * @brief x: The first passed 2D vector
 * @brief y: the second passed 2D vector
 * @return: A 2d vector with x.size() x y[0].size() dimensions
 */
template <typename T>
std::vector<std::vector<T>> multiply(std::vector<std::vector<T>> const& x,
                                     std::vector<std::vector<T>> const& y) {
    assert(x[0].size() == y.size());
    std::vector<std::vector<T>> out(x.size(), std::vector<T>(y[0].size()));

    for (size_t i = 0; i < x.size(); i++) {
        for (size_t j = 0; j < y[0].size(); j++) {
            for (size_t k = 0; k < x[0].size(); k++) {
                out[i][j] = x[i][k] * y[k][j];
            }
        }
    }

    return out;
}

template <typename T>
double multiply(std::vector<T> const& x, std::vector<T> const& y) {
    assert(x.size() == y.size());
    double out = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        out += x[i] * y[i];
    }

    return out;
}
