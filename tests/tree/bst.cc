#define ENABLE_TREE_VISUALIZATION
#include "../../src/classes/tree/bst.h"
#include "../../third_party/catch.hpp"
#include <string>

TEST_CASE("checking insertions and traversals") {
    bst<int> b1;

    b1.insert(2);
    b1.insert(19);
    b1.insert(5);

    std::vector<int> v = {2, 5, 19};
    std::vector<int> inorder = b1.inorder();
    REQUIRE(v == inorder);
}

TEST_CASE("checking removals in bst") {
    bst<int> b1;
    b1.insert(10);
    b1.insert(2);
    b1.insert(190);
    b1.insert(-20);
    b1.insert(23);

    b1.remove(190);
    std::vector<int> v = {-20, 2, 10, 23};
    std::vector<int> inorder = b1.inorder();
    REQUIRE(v == inorder);

    bst<int> b2;
    REQUIRE(b2.size() == 0);
    b2.remove(10);
    REQUIRE(b2.size() == 0);
    b2.insert(5);
    REQUIRE(b2.size() == 1);
    b2.remove(10);
    REQUIRE(b2.size() == 1);
    b2.remove(5);
    REQUIRE(b2.size() == 0);
}

TEST_CASE("checking search in bst") {
    bst<int> b;
    b.insert(1);
    b.insert(10);
    b.insert(-5);
    b.insert(8);
    REQUIRE(b.search(-5) == true);
    REQUIRE(b.search(-8) == false);

    bst<std::string> b1;
    b1.insert("hello");
    b1.insert("i'm");
    b1.insert("spiros");
    b1.insert("maggioros");
    REQUIRE(b1.search("spiros") == true);
    REQUIRE(b1.search("im") == false);
}

TEST_CASE("checking inorder in bst") {
    bst<char> b2;
    b2.insert('g');
    b2.insert('k');
    b2.insert('a');

    std::vector<char> _v = {'a', 'g', 'k'};
    std::vector<char> _inorder = b2.inorder();
    REQUIRE(_v == _inorder);

    bst<std::string> b3;
    b3.insert("hello");
    b3.insert("abraham");
    b3.insert("cheers");
    std::vector<std::string> __v = {"abraham", "cheers", "hello"};
    std::vector<std::string> __inorder = b3.inorder();
    REQUIRE(__v == __inorder);
}

TEST_CASE("checking preorder in bst") {
    bst<char> b;
    b.insert('g');
    b.insert('a');
    b.insert('b');
    b.insert('w');
    std::vector<char> pre = {'g', 'a', 'b', 'w'};
    REQUIRE(b.preorder() == pre);
}

TEST_CASE("checking postorder in bst") {
    bst<char> b;
    b.insert('g');
    b.insert('a');
    b.insert('b');
    b.insert('w');
    std::vector<char> post = {'b', 'a', 'w', 'g'};
    REQUIRE(b.postorder() == post);
}

TEST_CASE("checking iterators in bst") {
    bst<int> b({5, 3, -10, 4, 15, 20});
    std::vector<int> els = b.inorder();
    std::vector<int> check;
    for (auto it = b.begin(); it != b.end(); it++) {
        check.push_back(*(it));
    }

    REQUIRE(els == check);

    bst<int> tree;
    tree.insert(3);
    tree.insert(2);
    tree.insert(1);
    tree.insert(4);
    std::vector<int> v;
    auto it = tree.end();
    it--;
    for (; it != tree.begin(); it--) {
        v.push_back(*(it));
    }
    v.push_back(*(it));
    std::vector<int> check3 = {4, 3, 2, 1};
    REQUIRE(v == check3);
}

TEST_CASE("testing clear function in bst") {
    bst<char> b({'a', 'w', 'd', 'g', 'u', 'l'});
    b.clear();
    std::vector<char> v = b.inorder();
    REQUIRE(v.size() == 0);
}

TEST_CASE("testing copy constructor for bst class") {
    bst<int> b({5, 3, -10, 4, 15, 20});
    bst<int> b2(b);
    REQUIRE(b.inorder() == b2.inorder());
    REQUIRE(b.preorder() == b2.preorder());
}

TEST_CASE("testing operator = for bst class") {
    bst<int> b({5, 3, -10, 4, 15, 20});
    bst<int> b2;
    b2 = b;
    REQUIRE(b.inorder() == b2.inorder());
    REQUIRE(b.preorder() == b2.preorder());
}

TEST_CASE("testing level order in bst") {
    bst<int> t;
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
    std::vector<std::vector<int>> sol = {{5}, {1, 6}, {2, 8}, {3, 7, 10}, {4, 9}};
    REQUIRE(produced == sol);
}

TEST_CASE("Testing opearator << for bst") {
    bst<int> t({1, 10, 5, 3, 9, 25});
    REQUIRE_NOTHROW(cout << t);

    bst<char> tt({'a', 'g', 'q', 'v', 'w'});
    REQUIRE_NOTHROW(cout << tt);
}

#define TREE_VISUALIZATION_H
#ifdef TREE_VISUALIZATION_H

TEST_CASE("Testing binary search tree visualization") {
    bst<int> t;
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

    bst<char> b;
    b.insert('g');
    b.insert('a');
    b.insert('b');
    b.insert('w');
    CHECK_NOTHROW(b.visualize());
}

#endif
