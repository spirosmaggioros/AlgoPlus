#include "../../../src/algorithms/searching/binary_search.h"
#include "../../../third_party/catch.hpp"

TEST_CASE("testing binary search") {
    std::vector<int> v = {-10, 10, 20, 30, 100};
    REQUIRE(bin_search(v, 0, v.size() - 1, -10) == 0);
    REQUIRE(bin_search(v, 0, v.size() - 1, 30) == 3);
    REQUIRE(bin_search(v, 0, v.size() - 1, -20) == -1);
    REQUIRE(bin_search(v, 0, v.size() - 1, 100) == v.size() - 1);
}

TEST_CASE("testing lower_bound search") {
    std::vector<int> v = {10, 10, 20, 30, 30, 30, 30, 100};
    REQUIRE(lower_bound(v, 0, v.size(), 10) == 0);
    REQUIRE(lower_bound(v, 0, v.size(), 30) == 3);
    REQUIRE(lower_bound(v, 0, v.size(), 35) == 7);
    REQUIRE(lower_bound(v, 0, v.size(), 105) == v.size());
}

TEST_CASE("testing upper_bound search") {
    std::vector<int> v = {10, 10, 20, 30, 30, 30, 30, 100};
    REQUIRE(upper_bound(v, 0, v.size(), 35) == 7);
    REQUIRE(upper_bound(v, 0, v.size(), 30) == 7);
    REQUIRE(upper_bound(v, 0, v.size(), -20) == 0);
    REQUIRE(upper_bound(v, 0, v.size(), 100) == v.size());
}
