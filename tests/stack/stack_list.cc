#include "../../src/classes/stack/stack_list.h"
#include "../../third_party/catch.hpp"
#include <string>

TEST_CASE("testing push in stack") {
    stack_list<int> s;
    s.push(10);
    s.push(11);
    s.push(12);
    s.push(13);
    std::vector<int> els = {10, 11, 12, 13};
    reverse(els.begin(), els.end());
    std::vector<int> check;
    for (auto it = s.begin(); it != s.end(); it++) {
        check.push_back(*(it));
    }
    REQUIRE(els == check);

    stack_list<std::string> s2;
    s2.push("world");
    s2.push("hello");
    REQUIRE(s2.top() == "hello");
    s2.pop();
    REQUIRE(s2.top() == "world");
}

TEST_CASE("testing pop in stack") {
    stack_list<int> s;
    s.push(10);
    s.push(11);
    s.push(12);
    s.push(13);
    s.pop();
    REQUIRE(s.top() == 12);
    s.pop();
    REQUIRE(s.top() == 11);
    s.pop();
    REQUIRE(s.top() == 10);
}

TEST_CASE("testing size in stack") {
    stack_list<int> s;
    s.push(10);
    s.push(11);
    s.push(12);
    s.push(13);
    REQUIRE(s.size() == 4);
}

TEST_CASE("testing copy constructor for stack list class") {
    stack_list<char> s;
    s.push('a');
    s.push('b');
    s.push('c');
    s.push('d');
    s.push('e');

    stack_list<char> s2(s);
    std::vector<char> v1, v2;
    for (auto it = s.begin(); it != s.end(); it++) {
        v1.push_back(*(it));
    }

    for (auto it = s2.begin(); it != s2.end(); it++) {
        v2.push_back(*(it));
    }

    REQUIRE(v1 == v2);
}

TEST_CASE("testing operator = for stack list class") {
    stack_list<char> s;
    s.push('a');
    s.push('b');
    s.push('c');
    s.push('d');
    s.push('e');

    stack_list<char> s2;
    s2 = s;
    std::vector<char> v1, v2;
    for (auto it = s.begin(); it != s.end(); it++) {
        v1.push_back(*(it));
    }

    for (auto it = s2.begin(); it != s2.end(); it++) {
        v2.push_back(*(it));
    }

    REQUIRE(v1 == v2);
}