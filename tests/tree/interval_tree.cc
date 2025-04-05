#define ENABLE_TREE_VISUALIZATION
#include "../../src/classes/tree/interval_tree.h"
#include "../../third_party/catch.hpp"
#include <string>

TEST_CASE("testing insertion in interval tree") {
    interval_tree<int> i;
    i.insert({20, 36});
    i.insert({3, 41});
    i.insert({29, 99});
    i.insert({0, 1});
    i.insert({10, 15});
    std::vector<std::pair<int, int>> els = {{0, 1}, {3, 41}, {10, 15}, {20, 36}, {29, 99}};
    REQUIRE(i.inorder() == els);
}

TEST_CASE("testing overlap in interval tree") {
    interval_tree<int> i;
    i.insert({20, 36});
    i.insert({3, 41});
    i.insert({29, 99});
    i.insert({0, 1});
    i.insert({10, 15});
    REQUIRE(i.overlap({20, 36}, {3, 41}) == true);
    REQUIRE(i.overlap({20, 36}, {10, 15}) == false);
}

TEST_CASE("testing remove in interval tree") {
    interval_tree<char> i({{'a', 'b'}, {'c', 'd'}, {'a', 'd'}});
    REQUIRE(i.search({'a', 'b'}) == true);
    i.remove({'a', 'b'});
    REQUIRE(i.search({'a', 'b'}) == false);
    i.remove({'w', 'w'});
    REQUIRE(i.search({'w', 'w'}) == false);
    REQUIRE(i.search({'c', 'd'}));
    REQUIRE(i.search({'a', 'd'}) == true);
}

TEST_CASE("testing iterators in interval tree") {
    interval_tree<int> i({{1, 3}, {5, 6}, {2, 4}, {9, 10}});
    std::vector<std::pair<int, int>> els = i.inorder();
    std::vector<std::pair<int, int>> check;
    for (auto it = i.begin(); it != i.end(); it++) {
        check.push_back(*(it));
    }
    REQUIRE(check == els);
}

TEST_CASE("testing copy constructor for interval_tree class") {
    interval_tree<int> i({{1, 3}, {5, 6}, {2, 4}, {9, 10}});
    interval_tree<int> i2(i);
    REQUIRE(i.inorder() == i2.inorder());
    REQUIRE(i.preorder() == i2.preorder());

    interval_tree<char> i3({{'a', 'b'}, {'c', 'd'}, {'a', 'd'}});
    interval_tree<char> i4(i3);
    REQUIRE(i3.inorder() == i4.inorder());
    REQUIRE(i3.preorder() == i4.preorder());
}

TEST_CASE("testing operator = for interval tree class") {
    interval_tree<int> i({{1, 3}, {5, 6}, {2, 4}, {9, 10}});
    interval_tree<int> i2;
    i2 = i;
    REQUIRE(i.inorder() == i2.inorder());
    REQUIRE(i.preorder() == i2.preorder());
}

TEST_CASE("testing level order in interval tree") {
    interval_tree<int> t;
    t.insert({15, 20});
    t.insert({8, 13});
    t.insert({1, 20});
    t.insert({12, 15});
    t.insert({19, 30});
    t.insert({17, 19});
    t.insert({22, 25});
    std::vector<std::vector<std::pair<int, int>>> produced = t.level_order();
    std::vector<std::vector<std::pair<int, int>>> sol = {
        {{15, 20}}, {{8, 13}, {19, 30}}, {{1, 20}, {12, 15}, {17, 19}, {22, 25}}};
    REQUIRE(produced == sol);
}

#define TREE_VISUALIZATION_H
#ifdef TREE_VISUALIZATION_H

TEST_CASE("Testing interval tree visualization") {
    interval_tree<int> t;
    t.insert({15, 20});
    t.insert({8, 13});
    t.insert({1, 20});
    t.insert({12, 15});
    t.insert({19, 30});
    t.insert({17, 19});
    t.insert({22, 25});
    CHECK_NOTHROW(t.visualize());

    interval_tree<char> i({{'a', 'b'}, {'c', 'd'}, {'a', 'd'}});
    CHECK_NOTHROW(i.visualize());
}

#endif
