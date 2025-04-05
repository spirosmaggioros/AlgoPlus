#include "../../../src/algorithms/dynamic_programming/knapsack_1d.h"
#include "../../../third_party/catch.hpp"

using namespace std;

TEST_CASE("testing knapsack") {
    std::vector<int> v = {1, 2, 5};
    int amount = 11;
    REQUIRE(min_cost(v, amount) == 3);

    v.clear();
    v = {2};
    amount = 3;
    REQUIRE(min_cost(v, amount) == -1);

    v.clear();
    v = {1};
    amount = 0;
    REQUIRE(min_cost(v, amount) == 0);
}
