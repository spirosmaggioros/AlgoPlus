#include "../../../src/algorithms/sorting/merge_sort.h"
#include "../../../third_party/catch.hpp"
#include <random>
#include <vector>

TEST_CASE("testing merge sort") {
    std::vector<int64_t> v;
    for (int i = 0; i < 5000; i++) {
        int random = rand() % 50000;
        v.push_back(i - random);
    }
    merge_sort(v.begin(), v.end());
    REQUIRE(std::is_sorted(v.begin(), v.end()) == true);
}
