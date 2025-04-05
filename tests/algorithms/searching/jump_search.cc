#include "../../../src/algorithms/searching/jump_search.h"
#include "../../../third_party/catch.hpp"

TEST_CASE("testing jump_search algo") {
    std::vector<int> v = {-10, 4, 7, 10, 14, 18, 25};
    REQUIRE(jump_search(v, 3) == -1);
    REQUIRE(jump_search(v, 7) == 2);
    REQUIRE(jump_search(v, 14) == 4);
}