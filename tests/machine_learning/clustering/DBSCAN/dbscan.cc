#include "../../../../src/machine_learning/clustering/DBSCAN/dbscan.h"
#include "../../../../third_party/catch.hpp"
#include <vector>

TEST_CASE("Testing clustering [1] DBSCAN") {
    std::vector<std::pair<double, double>> v = {{1.0, 1.25},   {1.1, 1.3}, {0.9, 0.85},
                                                {40.0, 41.0},  {41.2, 42}, {39.8, 39.2},
                                                {100.4, 100.2}};
    DBSCAN a(v, 3, 1);
    std::map<std::pair<double, double>, int64_t> ans = a.get_clusters();
    std::map<std::pair<double, double>, int64_t> check;
    for (int i = 0; i < 3; i++) {
        check[v[i]] = 0;
    }
    for (int i = 3; i <= 5; i++) {
        check[v[i]] = 1;
    }
    check[v[6]] = 2;

    for (auto& x : ans) {
        REQUIRE(x.second == check[x.first]);
    }
}

TEST_CASE("Testing noise in [1] DBSCAN") {
    std::vector<std::pair<double, double>> v = {{1.0, 1.25}, {1.1, 1.3}, {0.9, 0.85}, {3, 3.5}};
    DBSCAN a(v, 0.5, 2);
    std::map<std::pair<double, double>, int64_t> ans = a.get_clusters();
    std::map<std::pair<double, double>, int64_t> check;
    for (int i = 0; i < 3; i++) {
        check[v[i]] = 0;
    }
    check[v[3]] = -1;

    for (auto& x : ans) {
        REQUIRE(x.second == check[x.first]);
    }
}
