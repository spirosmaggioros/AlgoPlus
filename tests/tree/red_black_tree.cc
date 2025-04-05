#define ENABLE_TREE_VISUALIZATION
#include "../../src/classes/tree/red_black_tree.h"
#include "../../third_party/catch.hpp"
#include <algorithm>
#include <string>

TEST_CASE("Testing red black tree insertion and traversal") {
    red_black_tree<int> rb;

    rb.insert(1);
    rb.insert(2);
    rb.insert(12);
    rb.insert(14);
    rb.insert(89);
    rb.insert(3);
    rb.insert(0);
    rb.insert(12);
    rb.insert(13);
    std::vector<int> expected_inorder = {0, 1, 2, 3, 12, 12, 13, 14, 89};
    std::vector<int> produced_inorder = rb.inorder();
    std::vector<int> expected_preorder = {12, 2, 1, 0, 3, 14, 12, 13, 89};
    std::vector<int> produced_preorder = rb.preorder();
    REQUIRE(expected_inorder == produced_inorder);
    REQUIRE(expected_preorder == produced_preorder);
}

TEST_CASE("Testing red black tree insertion, remove and traversal") {
    red_black_tree<int> rb;

    rb.insert(1);
    rb.insert(2);
    rb.insert(12);
    rb.insert(14);
    rb.insert(89);
    rb.insert(3);
    rb.insert(0);
    rb.insert(12);
    rb.insert(13);
    rb.remove(12);
    rb.remove(14);
    std::vector<int> expected_inorder = {0, 1, 2, 3, 12, 13, 89};
    std::vector<int> produced_inorder = rb.inorder();
    std::vector<int> expected_preorder = {12, 2, 1, 0, 3, 89, 13};
    std::vector<int> produced_preorder = rb.preorder();
    REQUIRE(expected_inorder == produced_inorder);
    REQUIRE(expected_preorder == produced_preorder);

    red_black_tree<char> t({'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'});
    std::vector<char> t_expected_inorder = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
    std::vector<char> t_produced_inorder = t.inorder();
    REQUIRE(t_expected_inorder == t_produced_inorder);
}

TEST_CASE("Testing search in red black tree") {
    red_black_tree<int> rb;

    rb.insert(1);
    rb.insert(2);
    rb.insert(12);
    rb.insert(14);
    rb.insert(89);
    rb.insert(3);
    rb.insert(0);
    rb.insert(12);
    rb.insert(13);
    rb.remove(12);
    rb.remove(14);
    REQUIRE(rb.search(0) == true);
    REQUIRE(rb.search(89) == true);
    REQUIRE(rb.search(14) == false);

    red_black_tree<char> t({'a', 'g', 'c', 'v', 'b', 'g'});
    REQUIRE(t.search('a') == true);
    REQUIRE(t.search('c') == true);
    t.remove('c');
    REQUIRE(t.search('c') == false);

    red_black_tree<std::string> tt({"hello", "we", "are", "csrt", "team", "from", "ntua"});
    REQUIRE(tt.search("are") == true);
    REQUIRE(tt.search("from") == true);
    tt.remove("are");
    tt.insert("CSRT");
    REQUIRE(tt.search("CSRT") == true);
}

TEST_CASE("Testing iterators in red black tree") {
    red_black_tree<int> rb({1, 2, 12, 14, 89, 3, 0, 13});
    std::vector<int> produced = rb.inorder();
    std::vector<int> check_1;
    for (auto it = rb.begin(); it != rb.end(); it++) {
        check_1.push_back(*(it));
    }

    std::vector<int> check_2;
    auto it = rb.end();
    it--;
    for (; it != rb.begin(); it--) {
        check_2.push_back(*(it));
    }
    check_2.push_back(*(it));
    REQUIRE(produced == check_1);
    std::reverse(produced.begin(), produced.end());
    REQUIRE(produced == check_2);

    red_black_tree<char> t({'a', 'c', 'd', 'e', 'f', 'g', 'l'});
    std::vector<char> t_produced = t.inorder();
    std::vector<char> t_check;
    for (auto it = t.begin(); it != t.end(); it++) {
        t_check.push_back(*(it));
    }

    REQUIRE(t_produced == t_check);
}

TEST_CASE("Testing clear function in red black tree") {
    red_black_tree<int> rb({1, 2, 3, 4, 5, 6});
    rb.clear();
    std::vector<int> produced = rb.inorder();
    REQUIRE(produced.size() == 0);
    REQUIRE(rb.size() == 0);

    red_black_tree<char> t({'a', 'g', 'e', 'l', 'm', 'd'});
    REQUIRE(t.size() == 6);
    t.clear();
    REQUIRE(t.size() == 0);
}

TEST_CASE("Testing size function in red black tree") {
    red_black_tree<int> rb({1, 2, 3, 4, 5, 6});
    REQUIRE(rb.size() == 6);

    red_black_tree<char> t({'a', 'c', 'e', 'g', 'd', 'f', 'u'});
    REQUIRE(t.size() == 7);
}

TEST_CASE("Testing operator= in red black tree") {
    red_black_tree<int> rb({1, 2, 3, 4, 5, 6});
    red_black_tree<int> rb2;
    rb2 = rb;
    REQUIRE(rb.inorder() == rb2.inorder());
    REQUIRE(rb.preorder() == rb2.preorder());
}

TEST_CASE("Testing operator== in red black tree") {
    red_black_tree<int> rb({1, 2, 3, 4, 5, 6});
    red_black_tree<int> rb2;
    rb2 = rb;
    REQUIRE(rb2 == rb);
}

TEST_CASE("Testing level order in red black tree") {
    red_black_tree<int> rb({1, 2, 3, 4, 5, 6});
    std::vector<std::vector<int>> produced = rb.level_order();
    std::vector<std::vector<int>> expected = {{2}, {1, 4}, {3, 5}, {6}};
    REQUIRE(produced == expected);
}

TEST_CASE("Testing level order in red black tree [2]") {
    red_black_tree<int> rb({10, 15, 25, 9, 8, 13, 45});
    std::vector<std::vector<int>> produced = rb.level_order();
    std::vector<std::vector<int>> expected = {{15}, {9, 25}, {8, 10, 45}, {13}};
    REQUIRE(produced == expected);
}

#define TREE_VISUALIZATION_H
#ifdef TREE_VISUALIZATION_H

TEST_CASE("Testing red black tree visualization") {
    red_black_tree<int> rb({10, 15, 25, 9, 8, 13, 45});
    CHECK_NOTHROW(rb.visualize());

    red_black_tree<char> t({'a', 'c', 'e', 'g', 'd', 'f', 'u'});
    CHECK_NOTHROW(t.visualize());
}

#endif
