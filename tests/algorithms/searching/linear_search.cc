#include "../../../src/algorithms/searching/linear_search.h"
#include "../../../third_party/catch.hpp"

TEST_CASE("testing linear search") {
    std::vector<int> v = {1, 5, 3, 1, 2, 3, -41, -20};
    REQUIRE(linear_search(v, 5) == true);
    REQUIRE(linear_search(v, -40) == false);
}