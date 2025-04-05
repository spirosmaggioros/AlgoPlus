#include "../../../src/algorithms/sorting/scheduling.h"
#include "../../../third_party/catch.hpp"

TEST_CASE("Testing scheduling [1]") {
    std::vector<std::pair<int, int>> v = {{0, 1}, {0, 2}, {1, 2}, {2, 4}};

    REQUIRE(scheduling(v) == 3);

    v.clear();
    v = {{0, 3}, {2, 5}, {3, 9}, {7, 8}};

    REQUIRE(scheduling(v) == 2);

    v.clear();
    v = {{1, 2}, {2, 3}, {4, 5}, {0, 1}};

    REQUIRE(scheduling(v) == 4);

    v.clear();
    v = {{0, 10}, {2, 3}, {4, 5}};

    REQUIRE(scheduling(v) == 2);

    v.clear();
    v = {{0, 4}, {3, 5}, {4, 8}};

    REQUIRE(scheduling(v) == 2);
}
