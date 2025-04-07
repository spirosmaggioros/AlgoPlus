#define ENABLE_LIST_VISUALIZATION
#include "../../src/classes/list/circular_linked_list.h"
#include "../../third_party/catch.hpp"
#include <string>

TEST_CASE("testing search in circular list") {
    // this is an example of a unit test to check if any algo or class works as it
    // should
    circular_linked_list<int> l1({1, 4, 5, 10});
    REQUIRE(l1.search(4) == true);
    REQUIRE(l1.search(150) == false);
}

TEST_CASE("push front in circular list") {
    circular_linked_list<int> l1;
    l1.push_front(1);
    l1.push_front(2);
    l1.push_front(5);
    std::vector<int> v = {5, 2, 1};
    std::vector<int> elem = l1.elements();
    REQUIRE(v == elem);
}

TEST_CASE("testing linked list with strings in circular list") {
    std::vector<std::string> __first = {"hello", "world"};
    circular_linked_list<std::string> l1(__first);
    REQUIRE(l1.search("hello") == true);
    l1.push_back("check");
    REQUIRE(l1.search("chec") == false); // needs to be fixed
}

TEST_CASE("testing linked list with chars in circular list") {
    std::vector<char> __first = {'a', 'b', 'c', 'd'};
    circular_linked_list<char> l1(__first);
    l1.erase('b');
    std::vector<char> __second = {'a', 'c', 'd'};
    std::vector<char> elem = l1.elements();
    REQUIRE(elem == __second);
    l1.push_back('g');
    __second.push_back('g');
    elem = l1.elements();
    REQUIRE(elem == __second);
}

TEST_CASE("testing iteration and visualization in circular list") {
    circular_linked_list<char> l1({'a', 'b', 'c', 'd'});
    std::vector<char> ans = {'a', 'b', 'c', 'd'};
    std::vector<char> elem = l1.elements();
    REQUIRE(elem == ans);
}

// TEST_CASE("testing reverse") {
//   circular_linked_list<int> l1;
//   l1.push_back(10);
//   l1.push_back(11);
//   l1.push_back(12);
//   l1.push_back(13);
//   l1.reverse();
//   std::vector<int> elem = l1.elements();
//   std::vector<int> v = {13, 12, 11, 10};
//   REQUIRE(elem == v);
//
//   circular_linked_list<std::string> l2({"hello", "world", "universe"});
//   l2.reverse();
//   std::vector<std::string> els = l2.elements();
//   std::vector<std::string> v2 = {"universe", "world", "hello"};
//   REQUIRE(els == v2);
// }

TEST_CASE("testing iterators in circular list") {
    circular_linked_list<int> l({4, 7, 1, 2, 3, 41, 32});
    std::vector<int> els = {4, 7, 1, 2, 3, 41, 32, 4, 7, 1, 2, 3, 41, 32};
    std::vector<int> v;
    int counter = 0;
    for (auto it = l.begin(); counter < 14 && it != l.end(); it++) {
        v.push_back(*(it));
        counter++;
    }
    REQUIRE(v == els);
}

TEST_CASE("testing copy constructors in circular list") {
    circular_linked_list<int> l({2, 4, 6, 7, 12, 9});
    circular_linked_list<int> l2(l);
    std::vector<int> v = l.elements();
    std::vector<int> v2 = l2.elements();
    REQUIRE(v == v2);
}

TEST_CASE("testing operator = in circular list") {
    circular_linked_list<int> l({2, 4, 6, 7, 12, 9});
    std::vector<int> v = l.elements();
    circular_linked_list<int> l2;
    l2 = l;
    std::vector<int> v2 = l2.elements();
    REQUIRE(v2 == v);
}

TEST_CASE("Testing circular linked list visualization") {
    circular_linked_list<int> l({2, 4, 6, 7, 12, 9});
    CHECK_NOTHROW(l.visualize());
}
