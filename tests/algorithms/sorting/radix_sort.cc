#include "../../../src/algorithms/sorting/radix_sort.h"
#include "../../../third_party/catch.hpp"
#include <algorithm>
#include <vector>

TEST_CASE("Test radix_sort function with integer vector") {
    std::vector<int> v = {170, 45, 75, 90, 24, 2, 66};
    radix_sort(v);
    REQUIRE(v == std::vector<int>{2, 24, 45, 66, 75, 90, 170});
}

TEST_CASE("Test radix_sort function with an already sorted vector") {
    std::vector<int> v = {1, 2, 3, 4, 5, 6};
    radix_sort(v);
    REQUIRE(std::is_sorted(v.begin(), v.end()) == true);
}
