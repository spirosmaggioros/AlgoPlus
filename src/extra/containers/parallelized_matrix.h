#ifndef PARALLELIZED_MATRIX_H
#define PARALLELIZED_MATRIX_H

#ifdef __cplusplus
#include <cassert>
#include <future>
#include <iostream>
#include <thread>
#include <vector>
#endif

int TOTAL_THREADS = 2;

namespace serial {
std::vector<std::vector<int64_t>> add(std::vector<std::vector<int64_t>>& v1,
                                      std::vector<std::vector<int64_t>>& v2) {
    if (v1.empty()) {
        return v2;
    }
    if (v2.empty()) {
        return v1;
    }

    int64_t n = std::min(v1.size(), v2.size()), m = std::min(v1[0].size(), v2[0].size());

    std::vector<std::vector<int64_t>> result(n, std::vector<int64_t>(m, 0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result[i][j] = v1[i][j] + v2[i][j];
        }
    }

    return result;
}
} // namespace serial

namespace parallel {

void compute(std::vector<std::vector<int64_t>>& result, std::vector<std::vector<int64_t>>& v1,
             std::vector<std::vector<int64_t>>& v2, const int64_t i, const int64_t j) {
    int64_t m = std::min(v1[0].size(), v2[0].size());

    for (int k = i; k < j; k++) {
        for (int w = 0; w < m; w++) {
            result[k][w] = v1[k][w] + v2[k][w];
        }
    }
}

std::vector<std::vector<int64_t>> add(std::vector<std::vector<int64_t>>& v1,
                                      std::vector<std::vector<int64_t>>& v2) {
    if (v1.empty()) {
        return v2;
    }
    if (v2.empty()) {
        return v1;
    }

    int64_t n = std::min(v1.size(), v2.size()), m = std::min(v1[0].size(), v2[0].size());

    std::vector<std::vector<int64_t>> result(n, std::vector<int64_t>(m, 0));

    std::vector<std::future<void>> threads;

    int batch_size = n / TOTAL_THREADS;
    for (int i = 0; i < TOTAL_THREADS; i++) {
        int start = i * batch_size;
        int end = (i == TOTAL_THREADS - 1) ? n : start + batch_size;
        threads.push_back(std::async(std::launch::async, compute, std::ref(result), std::ref(v1),
                                     std::ref(v2), start, end));
    }

    return result;
}
} // namespace parallel

#endif
