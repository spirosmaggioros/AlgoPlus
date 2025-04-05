#include "../../../src/algorithms/dynamic_programming/partitions.h"
#include "../../../third_party/catch.hpp"

TEST_CASE("Testing partition with 0") {
    std::vector<std::vector<long long>> part{partitions(0)};
    REQUIRE(part.size() == 1);
}

TEST_CASE("Testing partition with 4") {
    std::vector<std::vector<long long>> part{partitions(4)};
    REQUIRE(part.size() == 5);
}

TEST_CASE("Testing partition with 1") {
    std::vector<std::vector<long long>> part{partitions(1)};
    REQUIRE(part.size() == 1);
}
