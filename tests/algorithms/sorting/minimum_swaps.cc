#include "../../../src/algorithms/sorting/minimum_swaps.h"
#include "../../../third_party/catch.hpp"
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

TEST_CASE("Testing minimum swaps [1]") {
    std::vector<int> a{3, 6, 4, 8};
    std::vector<int> b{4, 6, 8, 3};

    REQUIRE(minimum_swaps(a, b) == 2);
}

TEST_CASE("Testing minimum swaps [2]") {
    std::vector<int> a{1, 2, 3, 4};
    std::vector<int> b{4, 3, 2, 1};

    REQUIRE(minimum_swaps(a, b) == 2);
}

TEST_CASE("Testing minimum swaps [3]") {
    std::vector<int> a{1, 2, 3, 4, 5, 6, 7};
    std::vector<int> b{1, 2, 3, 4, 5, 6, 7};

    REQUIRE(minimum_swaps(a, b) == 0);
}

TEST_CASE("Testing minimum swaps [4]") {
    std::vector<int> a(100), b(100);
    std::iota(a.begin(), a.end(), 1);
    std::iota(b.begin(), b.end(), 1);

    std::swap(a[0], a[99]);
    REQUIRE(minimum_swaps(a, b) == 1);
}
