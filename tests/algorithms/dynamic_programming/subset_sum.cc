#include "../../../src/algorithms/dynamic_programming/subset_sum.h"
#include "../../../third_party/catch.hpp"

TEST_CASE("Testing subset sum [1]") {
    std::vector<int> v{4, 2, 1, 5, 7};
    int B = 7;

    REQUIRE(subset_sum(v, B) == 3);
    v.clear();

    REQUIRE(subset_sum(v, B) == 0);
}

TEST_CASE("Testing subset sum [2]") {
    std::vector<int> v{1, 4, 6, 9};
    int B = 5;

    REQUIRE(subset_sum(v, B) == 1);
    B = 6;
    REQUIRE(subset_sum(v, 6) == 1);
    B = 9;
    REQUIRE(subset_sum(v, B) == 1);
}
