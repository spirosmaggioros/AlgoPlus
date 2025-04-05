#include "../../../src/algorithms/rmq/rmq_sparse_table.h"
#include "../../../third_party/catch.hpp"

TEST_CASE("Testing rmq 1") {
    std::vector<int> v{1, 5, 4, 2, 3, 7};
    RMQ<int> rr(v);

    REQUIRE(rr.query_value(0, 4) == 1);
    REQUIRE(rr.query_value(0, 3) == 1);
    REQUIRE(rr.query_value(0, 2) == 1);
}

TEST_CASE("Testing rmq 2") {
    std::vector<int> v{-1, -2, -3, -4, -5, -6};

    RMQ<int> rr(v);
    REQUIRE(rr.query_value(0, 1) == -1);
    REQUIRE(rr.query_value(0, 2) == -2);
    REQUIRE(rr.query_value(0, 3) == -3);
}
