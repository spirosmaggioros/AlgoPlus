#include "../../../src/algorithms/dynamic_programming/kadane.h"
#include "../../../third_party/catch.hpp"

TEST_CASE("testing kadane's algo") {
    std::vector<int> v = {1, 4, 5, -5, -6, 4, 10, 15, -10};
    REQUIRE(kadane(v) == 29);

    v.clear();
    v = {1};
    REQUIRE(kadane(v) == 1);

    v.clear();
    v = {1, 4, 5, 6, 7, 8};
    REQUIRE(kadane(v) == 31);
}