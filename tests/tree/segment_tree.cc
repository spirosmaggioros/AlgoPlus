#include "../../src/classes/tree/segment_tree.h"
#include "../../third_party/catch.hpp"
#include "../../src/classes/tree/segment_tree_iterative.h"

TEST_CASE("Testing default constructor of the segment tree") {
    std::vector<int> v = {1, 5, 2, 4};
    segment_tree t(v);
    seg_tree<int> tt(v);

    REQUIRE(t.sum(0, 1) == 6);
    REQUIRE(t.sum(0, 5) == -1);
    REQUIRE(tt.sum(0, 1) == 6);
}

TEST_CASE("Testing update of the segment tree") {
    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8};
    segment_tree t(v);
    seg_tree<int> tt(v);

    REQUIRE(t.sum(0, 2) == 6);
    REQUIRE(tt.sum(0, 2) == 6);
    t.update(0, 0);
    tt.update(0, 0);
    REQUIRE(t.sum(0, 2) == 5);
    REQUIRE(tt.sum(0, 2) == 5);
    t.update(0, -5);
    tt.update(0, -5);
    REQUIRE(t.sum(0, 2) == 0);
    REQUIRE(tt.sum(0, 2) == 0);
    for (int i = 0; i < v.size(); i++) {
        t.update(i, 0);
        tt.update(i, 0);
    }

    REQUIRE(t.sum(0, v.size() - 1) == 0);
    REQUIRE(tt.sum(0, v.size() - 1) == 0);
}

TEST_CASE("Testing segment tree queries [2]") {
    std::vector<int> v = {-10, -20, -30, -40, -50};
    segment_tree t(v);
    seg_tree<int> tt(v);
    REQUIRE(t.sum(0, 2) == -60);
    REQUIRE(tt.sum(0, 2) == -60);
    REQUIRE(t.sum(v.size() - 2, v.size() - 1) == -90);
    REQUIRE(tt.sum(v.size() - 2, v.size() - 1) == -90);
    for (int i = 0; i < v.size(); i++) {
        t.update(i, i);
        tt.update(i, i);
    }
    REQUIRE(t.sum(0, 2) == (2 * 3) / 2);
    REQUIRE(tt.sum(0, 2) == (2 * 3) / 2);
    REQUIRE(t.sum(0, 4) == (4 * 5) / 2);
    REQUIRE(tt.sum(0, 4) == (4 * 5) / 2);
}
