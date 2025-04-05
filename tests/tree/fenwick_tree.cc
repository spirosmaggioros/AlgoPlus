#include "../../src/classes/tree/fenwick_tree.h"
#include "../../third_party/catch.hpp"

TEST_CASE("Testing fenwick tree default constructor") {
    std::vector<int> v = {1, 4, 5, 6, 7};
    CHECK_NOTHROW(fenwick_tree<int>(v));
}

TEST_CASE("Testing fenwick tree sum query [1]") {
    std::vector<int> v = {1, 2, 3, 4, 5};
    fenwick_tree<int> f(v);
    REQUIRE(f.sum(2) == 6);
    REQUIRE(f.sum(1) == 3);
    REQUIRE(f.sum(4) == 15);
    REQUIRE(f.sum(0) == 1);
}

TEST_CASE("Testing fenwick tree sum query [2]") {
    std::vector<int> v = {1, 2, 3, 4, 5};
    fenwick_tree<int> f(v);
    REQUIRE(f.sum(2) == 6);
    REQUIRE(f.sum(4) == 15);
    REQUIRE(f.sum(1, 2) == 5);
    REQUIRE(f.sum(3, 4) == 9);
    REQUIRE(f.sum(1, 3) == 9);
}

TEST_CASE("Testing update query [1]") {
    std::vector<int> v = {1, 2, 3, 4, 5};
    fenwick_tree<int> f(v);

    f.update(0, 2);
    REQUIRE(f.sum(0, 1) == 5);
    f.update(0, -5);
    REQUIRE(f.sum(0, 2) == 3);
}
