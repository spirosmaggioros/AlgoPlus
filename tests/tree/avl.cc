#include "../../third_party/catch.hpp"
#include <iostream>
#include <string>

#define ENABLE_TREE_VISUALIZATION
#include "../../src/classes/tree/avl_tree.h"

TEST_CASE("checking insertions and traversals in avl") {
    avl_tree<int> a1;

    a1.insert(10);
    a1.insert(15);
    a1.insert(4);
    a1.insert(-150);
    std::vector<int> v = {-150, 4, 10, 15};
    std::vector<int> inorder = a1.inorder();
    REQUIRE(v == inorder);
}

TEST_CASE("checking search in avl") {
    avl_tree<std::string> a;
    a.insert("hello");
    a.insert("world");
    a.insert("i'm");
    a.insert("spirosmag");
    REQUIRE(a.search("world") == true);
    REQUIRE(a.search("im") == false);
}

TEST_CASE("checking removals in avl") {
    avl_tree<int> a1;
    a1.insert(10);
    a1.insert(2);
    a1.insert(190);
    a1.insert(-20);
    a1.insert(23);
    a1.remove(190);
    std::vector<int> v = {-20, 2, 10, 23};
    std::vector<int> inorder = a1.inorder();
    REQUIRE(v == inorder);
}

TEST_CASE("checking inorder in avl") {
    avl_tree<char> a2;
    a2.insert('g');
    a2.insert('d');
    a2.insert('w');
    a2.insert('o');
    std::vector<char> _v = {'d', 'g', 'o', 'w'};
    std::vector<char> _inorder = a2.inorder();
    REQUIRE(_v == _inorder);

    avl_tree<std::string> a3;
    a3.insert("hello");
    a3.insert("from");
    a3.insert("mars");
    a3.insert("cheers");
    std::vector<std::string> __v = {"cheers", "from", "hello", "mars"};
    std::vector<std::string> __inorder = a3.inorder();
    REQUIRE(__v == __inorder);
}

TEST_CASE("checking preorder in avl") {
    avl_tree<int> a;
    a.insert(10);
    a.insert(5);
    a.insert(-5);
    a.insert(2);
    std::vector<int> __preorder = {5, -5, 2, 10};
    REQUIRE(a.preorder() == __preorder);
}

TEST_CASE("checking postorder in avl") {
    avl_tree<int> a;
    a.insert(10);
    a.insert(5);
    a.insert(-5);
    a.insert(2);
    std::vector<int> __postorder = {-5, 2, 10, 5};
    REQUIRE(a.postorder() == __postorder);
}

TEST_CASE("testing iterators in avl") {
    avl_tree<int> a({8, 4, 3, 1, 5, 6});
    std::vector<int> els = a.inorder();
    std::vector<int> check;
    for (auto it = a.begin(); it != a.end(); it++) {
        check.push_back(*(it));
    }

    REQUIRE(check == els);

    avl_tree<char> b({'a', 'g', 'd', 'w', 'v', 's'});
    std::vector<char> els2 = b.inorder();
    std::vector<char> check2;
    for (auto it = b.begin(); it != b.end(); it++) {
        check2.push_back(*(it));
    }

    REQUIRE(check2 == els2);

    avl_tree<int> tree;
    tree.insert(1);
    tree.insert(2);
    tree.insert(3);
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

TEST_CASE("testing clear function in avl") {
    avl_tree<char> a({'g', 'w', 'h', 'p', 'u'});
    a.clear();
    std::vector<char> v = a.inorder();
    REQUIRE(v.size() == 0);
    a.insert('g');
    a.insert('w');
    a.insert('a');
    std::vector<char> v2 = a.inorder();
    std::vector<char> check = {'a', 'g', 'w'};
    REQUIRE(v2 == check);
}

TEST_CASE("testing copy constructor for avl tree class") {
    avl_tree<int> a({8, 4, 3, 1, 5, 6});
    avl_tree<int> b(a);
    REQUIRE(a.inorder() == b.inorder());
    REQUIRE(a.preorder() == b.preorder());
}

TEST_CASE("testing operator = for avl tree class") {
    avl_tree<int> a({8, 4, 3, 1, 5, 6});
    avl_tree<int> b;
    b = a;
    REQUIRE(a.inorder() == b.inorder());
    REQUIRE(a.preorder() == b.preorder());
}

TEST_CASE("testing level order in avl tree") {
    avl_tree<int> t;
    for (int i = 1; i <= 10; i++)
        t.insert(i);
    std::vector<std::vector<int>> produced = t.level_order();
    std::vector<std::vector<int>> sol = {{4}, {2, 8}, {1, 3, 6, 9}, {5, 7, 10}};
    REQUIRE(produced == sol);
}

TEST_CASE("Testing get_root function in avl tree") {
    avl_tree<int> t;
    t.insert(35);
    t.insert(30);
    t.insert(38);
    t.insert(22);
    t.insert(36);
    t.insert(45);

    REQUIRE(t.get_root() == 35);
    t.remove(35);
    REQUIRE(t.get_root() == 36);
}

TEST_CASE("Testing operator << for avl tree") {
    avl_tree<int> t({1, 10, 2, 8, 6, 5});
    REQUIRE_NOTHROW(std::cout << t);

    avl_tree<char> tt({'a', 'g', 'd', 'z', 'w', 'f'});
    REQUIRE_NOTHROW(std::cout << tt);
}

TEST_CASE("Testing avl tree visualization") {
    avl_tree<int> t;
    t.insert(35);
    t.insert(30);
    t.insert(38);
    t.insert(22);
    t.insert(36);
    t.insert(45);

    CHECK_NOTHROW(t.visualize());

    avl_tree<char> a({'g', 'w', 'h', 'p', 'u'});
    CHECK_NOTHROW(a.visualize());
}
