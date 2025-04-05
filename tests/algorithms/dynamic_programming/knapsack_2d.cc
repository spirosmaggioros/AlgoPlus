#include "../../../src/algorithms/dynamic_programming/knapsack_2d.h"
#include "../../../third_party/catch.hpp"

TEST_CASE("Testing knapsack 0/1 [1]") {
    std::vector<std::pair<int, int>> v = {{10, 60}, {20, 100}, {30, 120}};

    REQUIRE(knapsack(v, 50) == 220);
    v.clear();
    v = {{5, 40}, {3, 20}, {6, 10}, {3, 30}};

    REQUIRE(knapsack(v, 10) == 70);
}
