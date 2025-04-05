#include "../../../src/algorithms/sorting/bucket_sort.h"
#include "../../../third_party/catch.hpp"
#include <random>
#include <string>

TEST_CASE("testing radix sort") {
    std::vector<float> v = {0.004, 0.004123, 0.03213, 0.1234, 0.42131, 0.0000123, 0.04313};
    bucket_sort(v);
    REQUIRE(std::is_sorted(v.begin(), v.end()) == true);
}
