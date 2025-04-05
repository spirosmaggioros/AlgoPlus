#include "../../third_party/catch.hpp"
#include <string>
#include "../../src/classes/tree/234_tree.h"

TEST_CASE("Testing 234-tree insertion [1]") {
    ttf_tree<int> t;
    t.insert(30);
    t.insert(99);
    t.insert(70);
    std::vector<std::vector<std::vector<int>>> level_ordered = t.level_order();
    std::vector<std::vector<std::vector<int>>> check = {{{30, 70, 99}}};
    REQUIRE(level_ordered == check);
}

TEST_CASE("Testing 234-tree insertion [2]") {
    ttf_tree<int> t;
    t.insert(30);
    t.insert(99);
    t.insert(70);
    t.insert(60);
    t.insert(40);
    std::vector<std::vector<std::vector<int>>> level_ordered = t.level_order();
    std::vector<std::vector<std::vector<int>>> check = {{{70}}, {{30, 40, 60}, {99}}};
    REQUIRE(level_ordered == check);
}

TEST_CASE("Testing 234-tree with default constructor [2]") {
    ttf_tree<int> t({30, 99, 70});

    std::vector<std::vector<std::vector<int>>> level_ordered = t.level_order();
    std::vector<std::vector<std::vector<int>>> check = {{{30, 70, 99}}};

    REQUIRE(level_ordered == check);
}

TEST_CASE("Testing 234-tree searching [1]") {
    ttf_tree<int> t;
    t.insert(30);
    t.insert(99);
    t.insert(70);
    t.insert(60);
    t.insert(40);
    REQUIRE(t.search(30) == true);
    REQUIRE(t.search(60) == true);
    REQUIRE(t.search(50) == false);
    REQUIRE(t.search(100) == false);
}

TEST_CASE("Testing 234-tree insertion [3]") {
    ttf_tree<int> t;
    t.insert(30);
    t.insert(99);
    t.insert(70);
    t.insert(60);
    t.insert(40);
    t.insert(66);
    t.insert(50);
    std::vector<std::vector<std::vector<int>>> level_ordered = t.level_order();
    std::vector<std::vector<std::vector<int>>> check = {{{40, 70}}, {{30}, {50, 60, 66}, {99}}};
    REQUIRE(level_ordered == check);
}

TEST_CASE("Testing 234-tree insertion [4]") {
    ttf_tree<int> t;
    t.insert(30);
    t.insert(99);
    t.insert(70);
    t.insert(60);
    t.insert(40);
    t.insert(66);
    t.insert(50);
    t.insert(53);
    t.insert(45);
    t.insert(42);
    std::vector<std::vector<std::vector<int>>> level_ordered = t.level_order();
    std::vector<std::vector<std::vector<int>>> check = {
        {{60}}, {{40, 50}, {70}}, {{30}, {42, 45}, {53}, {66}, {99}}};
    REQUIRE(level_ordered == check);
}
