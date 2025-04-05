#include "../../src/classes/list/skip_list.h"
#include "../../third_party/catch.hpp"
#include <string>

TEST_CASE("testing search in skip list") {
    skip_list<int> s(3, 0.5);
    s.insert(3);
    s.insert(6);
    s.insert(7);
    s.insert(9);
    s.insert(12);
    s.insert(19);
    s.insert(17);
    s.insert(15);
    s.insert(100);
    s.insert(500);
    s.insert(174);
    s.insert(1234);
    // std::cout << s << '\n';
    REQUIRE(s.search(19) == true);
    REQUIRE(s.search(12) == true);
    REQUIRE(s.search(1234) == true);
}

TEST_CASE("testing deletion in skip list") {
    skip_list<char> s(3, 0.5);
    s.insert('a');
    s.insert('b');
    s.insert('d');
    s.insert('r');
    REQUIRE(s.search('b') == true);
    // std::cout << s << '\n';
    s.remove('b');
    // std::cout << s << '\n';
    REQUIRE(s.search('b') == false);
}

TEST_CASE("testing iterators in skip list") {
    skip_list<int> s(3, 0.5);
    s.insert(10);
    s.insert(11);
    s.insert(12);
    s.insert(13);
    s.insert(14);
    s.insert(-10);
    std::vector<int> els = {-10, 10, 11, 12, 13, 14};
    std::vector<int> v;
    for (auto it = s.begin(); it != s.end(); it++) {
        v.push_back(*it);
    }
    REQUIRE(els == v);
}

TEST_CASE("testing copy constructors in skip list") {
    skip_list<char> s(3, 0.5);
    s.insert('a');
    s.insert('b');
    s.insert('d');
    s.insert('r');

    skip_list<char> s2(s);
    std::vector<char> v1, v2;
    for (auto it = s.begin(); it != s.end(); it++) {
        v1.push_back(*(it));
    }
    for (auto it = s2.begin(); it != s2.end(); it++) {
        v2.push_back(*(it));
    }

    REQUIRE(v1 == v2);
}
