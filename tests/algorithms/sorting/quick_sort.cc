#include "../../../src/algorithms/sorting/quick_sort.h"
#include "../../../third_party/catch.hpp"
#include <random>
#include <vector>

TEST_CASE("testing quick sort") {
    std::vector<int64_t> v;
    for (int i = 0; i < 5000; i++) {
        int random = rand() % 50000;
        v.push_back(i - random);
    }
    quick_sort(v.begin(), v.end());
    REQUIRE(std::is_sorted(v.begin(), v.end()) == true);
}

TEST_CASE("testing quick sort 2") {
    std::vector<char> v;
    std::vector<char> characters = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                                    'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                                    's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
    for (int i = 0; i < 10000; i++) {
        int random = rand() % 25;
        v.push_back(characters[random]);
    }
    quick_sort(v.begin(), v.end());
    REQUIRE(std::is_sorted(v.begin(), v.end()) == true);
}
