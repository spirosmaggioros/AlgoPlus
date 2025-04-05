#define ENABLE_TREE_VISUALIZATION
#include "../../src/classes/tree/splay_tree.h"
#include "../../third_party/catch.hpp"
#include <string>

TEST_CASE("testing insertion in splay tree") {
    splay_tree<int> s;
    s.insert(10);
    s.insert(12);
    s.insert(16);
    s.insert(5);
    s.insert(-10);
    s.insert(6);
    std::vector<int> els = s.inorder();
    std::vector<int> check = {-10, 5, 6, 10, 12, 16};
    REQUIRE(els == check);
}

TEST_CASE("testing remove in splay tree") {
    splay_tree<int> s;
    s.insert(10);
    s.insert(12);
    s.insert(16);
    s.insert(5);
    s.insert(-10);
    s.insert(6);
    REQUIRE(s.search(6) == true);
    REQUIRE(s.search(16) == true);
    REQUIRE(s.search(15) == false);
    s.remove(6);
    REQUIRE(s.search(6) == false);
    s.remove(16);
    REQUIRE(s.search(16) == false);
    s.insert(15);
    REQUIRE(s.search(15) == true);
}

TEST_CASE("testing size in splay tree") {
    splay_tree<int> s;
    s.insert(10);
    s.insert(12);
    s.insert(16);
    s.insert(5);
    s.insert(-10);
    s.insert(6);
    REQUIRE(s.size() == 6);
    s.remove(10);
    REQUIRE(s.size() == 5);

    splay_tree<std::string> s2({"hello", "there", "universe", "we", "are", "here"});
    REQUIRE(s2.size() == 6);
}

TEST_CASE("testing clear function in splay tree") {
    splay_tree<int> s;
    s.insert(10);
    s.insert(12);
    s.insert(16);
    s.insert(5);
    s.insert(-10);
    s.insert(6);
    s.clear();
    std::vector<int> els = s.inorder();
    REQUIRE(els.size() == 0);
}

TEST_CASE("testing iterators in splay tree") {
    splay_tree<int> s;
    s.insert(10);
    s.insert(12);
    s.insert(16);
    s.insert(5);
    s.insert(-10);
    s.insert(6);
    std::vector<int> v = {-10, 5, 6, 10, 12, 16};
    std::vector<int> check;
    for (auto it = s.begin(); it != s.end(); it++) {
        check.push_back(*(it));
    }
    REQUIRE(check == v);

    splay_tree<int> tree;
    tree.insert(1);
    tree.insert(4);
    tree.insert(2);
    tree.insert(3);
    std::vector<int> v2;
    auto it = tree.end();
    it--;
    for (; it != tree.begin(); it--) {
        v2.push_back(*(it));
    }
    v2.push_back(*(it));
    std::vector<int> check3 = {4, 3, 2, 1};
    REQUIRE(v2 == check3);
}

TEST_CASE("testing copy constructor for splay tree class") {
    splay_tree<char> s({'a', 'w', 'g', 'o', 's', 'v'});
    splay_tree<char> s2(s);
    REQUIRE(s.inorder() == s2.inorder());
    REQUIRE(s.preorder() == s2.preorder());
}

TEST_CASE("testing operator = for splay tree class") {
    splay_tree<char> s({'a', 'w', 'g', 'o', 's', 'v'});
    splay_tree<char> s2;
    s2 = s;
    REQUIRE(s.inorder() == s2.inorder());
    REQUIRE(s.preorder() == s2.preorder());
}

TEST_CASE("testing level order for splay tree") {
    splay_tree<int> t;
    t.insert(5);
    t.insert(1);
    t.insert(2);
    t.insert(3);
    t.insert(6);
    t.insert(8);
    t.insert(7);
    t.insert(10);
    t.insert(9);
    t.insert(4);
    std::vector<std::vector<int>> produced = t.level_order();
    std::vector<std::vector<int>> sol = {{4}, {3, 8}, {2, 6, 9}, {1, 5, 7, 10}};
    REQUIRE(produced == sol);
}

TEST_CASE("Testing operator << for splay tree") {
    splay_tree<int> t({1, 5, 3, 2, 9, 11});
    REQUIRE_NOTHROW(cout << t);

    splay_tree<char> tt({'a', 'b', 'w', 'z', 'f', 'd'});
    REQUIRE_NOTHROW(cout << tt);
}

#define TREE_VISUALIZATION_H
#ifdef TREE_VISUALIZATION_H

TEST_CASE("Testing splay tree visualization") {
    splay_tree<int> t;
    t.insert(5);
    t.insert(1);
    t.insert(2);
    t.insert(3);
    t.insert(6);
    t.insert(8);
    t.insert(7);
    t.insert(10);
    t.insert(9);
    t.insert(4);
    CHECK_NOTHROW(t.visualize());

    splay_tree<char> s({'a', 'w', 'g', 'o', 's', 'v'});
    CHECK_NOTHROW(s.visualize());
}

#endif
