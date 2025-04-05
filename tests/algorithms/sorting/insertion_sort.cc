#include "../../../src/algorithms/sorting/insertion_sort.h"
#include "../../../third_party/catch.hpp"
#include <random>
#include <string>

TEST_CASE("testing insertion sort") {
    std::vector<int64_t> v;
    for (int i = 0; i < 5000; i++) {
        int random = rand() % 50000;
        v.push_back(i - random);
    }
    insertion_sort(v);
    REQUIRE(std::is_sorted(v.begin(), v.end()) == true);
}

TEST_CASE("testing insertion sort v2") {
    std::vector<std::string> v = {"sdkjsad", "9jssdasd8", "odosadoau0923",
                                  "08asdh",  "lsda02",    "as0a9se"};
    insertion_sort(v);
    REQUIRE(std::is_sorted(v.begin(), v.end()) == true);
}
