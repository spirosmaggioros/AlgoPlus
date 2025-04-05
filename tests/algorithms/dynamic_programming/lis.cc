#include "../../../src/algorithms/dynamic_programming/lis.h"
#include "../../../third_party/catch.hpp"

TEST_CASE("testing longest increasing subsequence") {
    std::vector<int> v = {1, 5, 6, 7};
    REQUIRE(lis(v) == 4);

    std::vector<char> s = {'a', 'b', 'g', 'e', 'f'};
    REQUIRE(lis(s) == 4);

    std::vector<std::string> w = {"hello", "world", "im", "coding"};
    REQUIRE(lis(w) == 2);

    v.clear();
    v = {};
    REQUIRE(lis(v) == 0);
}