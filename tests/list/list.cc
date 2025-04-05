#define ENABLE_LIST_VISUALIZATION
#include "../../third_party/catch.hpp"
#include <string>
#include "../../src/classes/list/linked_list.h"

TEST_CASE("testing search in single list") {
    // this is an example of a unit test to check if any algo or class works as it
    // should
    linked_list<int> l1({1, 4, 5, 10});
    REQUIRE(l1.search(4) == true);
    REQUIRE(l1.search(150) == false);
}

TEST_CASE("push front in single list") {
    linked_list<int> l1;
    l1.push_front(1);
    l1.push_front(2);
    l1.push_front(5);
    std::vector<int> v = {5, 2, 1};
    std::vector<int> elem = l1.elements();
    REQUIRE(v == elem);
}

TEST_CASE("testing linked list with strings in single list") {
    std::vector<std::string> __first = {"hello", "world"};
    linked_list<std::string> l1(__first);
    REQUIRE(l1.search("hello") == true);
    l1.push_back("check");
    REQUIRE(l1.search("chec") == false); // needs to be fixed
}

TEST_CASE("testing linked list with chars in single list") {
    std::vector<char> __first = {'a', 'b', 'c', 'd'};
    linked_list<char> l1(__first);
    l1.erase('b');
    std::vector<char> __second = {'a', 'c', 'd'};
    std::vector<char> elem = l1.elements();
    REQUIRE(elem == __second);
    l1.push_back('g');
    __second.push_back('g');
    elem = l1.elements();
    REQUIRE(elem == __second);
}

TEST_CASE("testing iteration in single list") {
    linked_list<char> l1({'a', 'b', 'c', 'd'});
    std::vector<char> ans = {'a', 'b', 'c', 'd'};
    std::vector<char> elem = l1.elements();
    REQUIRE(elem == ans);
}

TEST_CASE("testing reverse in single list") {
    linked_list<int> l1;
    l1.push_back(10);
    l1.push_back(11);
    l1.push_back(12);
    l1.push_back(13);
    l1.reverse();
    std::vector<int> elem = l1.elements();
    std::vector<int> v = {13, 12, 11, 10};
    REQUIRE(elem == v);

    linked_list<std::string> l2({"hello", "world", "universe"});
    l2.reverse();
    std::vector<std::string> els = l2.elements();
    std::vector<std::string> v2 = {"universe", "world", "hello"};
    REQUIRE(els == v2);
}

TEST_CASE("testing iterators in single list") {
    linked_list<int> l({4, 7, 1, 2, 3, 41, 32});
    std::vector<int> els = l.elements();
    std::vector<int> v;
    for (auto it = l.begin(); it != l.end(); it++) {
        v.push_back(*(it));
    }
    REQUIRE(v == els);
}

TEST_CASE("testing copy constructors in single list") {
    linked_list<int> l({2, 5, 46, 2, 8, 5});
    linked_list<int> l2(l);
    std::vector<int> v = l.elements();
    std::vector<int> v2 = l2.elements();
    REQUIRE(v == v2);
}

TEST_CASE("testing operator = in single list") {
    linked_list<int> l({2, 5, 46, 2, 8, 5});
    std::vector<int> v = l.elements();

    linked_list<int> l2;
    l2 = l;
    std::vector<int> v2 = l2.elements();
    REQUIRE(v2 == v);
}

TEST_CASE("Bug with tail when pushing front and then pushing back") {
    linked_list<char> l;
    l.push_front('d');
    l.push_front('c');
    l.push_front('b');
    l.push_front('a');
    l.push_back('e');
    l.push_back('f');

    std::vector<char> v = l.elements();
    std::vector<char> check = {'a', 'b', 'c', 'd', 'e', 'f'};
    REQUIRE(v == check);
}

TEST_CASE("Testing linked list visualization") {
    linked_list<char> l;
    l.push_front('d');
    l.push_front('c');
    l.push_front('b');
    l.push_front('a');
    l.push_back('e');
    l.push_back('f');

    CHECK_NOTHROW(l.visualize());
}
