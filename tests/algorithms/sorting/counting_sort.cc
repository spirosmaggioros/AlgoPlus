#include "../../../src/algorithms/sorting/counting_sort.h"
#include "../../../third_party/catch.hpp"
#include <algorithm>
#include <vector>

TEST_CASE("Test counting_sort function with integer vector") {
    std::vector<int> v = {5, 4, 3, 2, 1};
    counting_sort(v);
    REQUIRE(v == std::vector<int>{1, 2, 3, 4, 5});
}

TEST_CASE("Test counting_sort function with another integer vector") {
    std::vector<int> v;
    for (int i = 10; i < 5000; i++) {
        int random = rand() % (i - 1);
        v.push_back(i - random);
    }
    counting_sort(v);
    REQUIRE(std::is_sorted(v.begin(), v.end()) == true);
}
