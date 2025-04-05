#include "../../../src/algorithms/sorting/heap_sort.h"
#include "../../../third_party/catch.hpp"
#include <random>
#include <string>

TEST_CASE("testing heap sort") {
    std::vector<int64_t> v;
    for (int i = 0; i < 5000; i++) {
        int random = rand() % 50000;
        v.push_back(i - random);
    }
    heap_sort(v);
    REQUIRE(std::is_sorted(v.begin(), v.end()) == true);
}
