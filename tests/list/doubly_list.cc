#define ENABLE_LIST_VISUALIZATION
#include "../../third_party/catch.hpp"
#include <string>
#include "../../src/classes/list/doubly_linked_list.h"

// TEST DOES NOT PASS
TEST_CASE("testing push back in doubly list") {
    doubly_linked_list<int> l1;
    doubly_linked_list<char> l2;
    doubly_linked_list<std::string> l3;

    l1.push_front(1);
    l1.push_front(4);
    l1.push_front(5);
    std::vector<int> v = {5, 4, 1};
    std::vector<int> elem = l1.elements();
    REQUIRE(v == elem);

    l2.push_back('a');
    l2.push_back('g');
    l2.push_back('s');
    std::vector<char> _v = {'a', 'g', 's'};
    std::vector<char> _elem = l2.elements();
    REQUIRE(_v == _elem);

    l3.push_back("hello");
    l3.push_back("from");
    l3.push_back("mars");
    std::vector<std::string> __v = {"hello", "from", "mars"};
    std::vector<std::string> __elem = l3.elements();
    REQUIRE(__v == __elem);
}

TEST_CASE("testing push front in doubly list") {
    doubly_linked_list<int> l1;
    doubly_linked_list<char> l2;
    doubly_linked_list<std::string> l3;
    l1.push_front(3);
    l1.push_front(5);
    l1.push_front(10);
    std::vector<int> v = {10, 5, 3};
    std::vector<int> elem = l1.elements();
    REQUIRE(v == elem);

    l2.push_front('a');
    l2.push_front('f');
    l2.push_front('j');
    std::vector<char> _v = {'j', 'f', 'a'};
    std::vector<char> _elem = l2.elements();
    REQUIRE(_v == _elem);

    l3.push_front("hello");
    l3.push_front("from");
    l3.push_front("mars");
    std::vector<std::string> __v = {"mars", "from", "hello"};
    std::vector<std::string> __elem = l3.elements();
    REQUIRE(__v == __elem);
}

TEST_CASE("testing search in doubly list") {
    doubly_linked_list<int> l1({3, 5, 7, 8});
    doubly_linked_list<char> l2({'a', 'v', 'd'});
    doubly_linked_list<std::string> l3({"hello", "there", "everyone"});

    REQUIRE(l1.search(3) == true);
    REQUIRE(l1.search(9) == false);
    REQUIRE(l2.search('a') == true);
    REQUIRE(l2.search('m') == false);
    REQUIRE(l3.search("hello") == true);
    REQUIRE(l3.search("hellos") == false);
}

TEST_CASE("testing erase in doubly list") {
    doubly_linked_list<int> l({4, 5, 6, 7, 8});
    REQUIRE(l.search(4) == true);
    REQUIRE(l.search(7) == true);
    l.erase(4);
    REQUIRE(l.search(4) == false);
    l.erase(7);
    REQUIRE(l.search(7) == false);
    l.erase(6);
    REQUIRE(l.search(6) == false);
}

TEST_CASE("testing reverse in doubly list") {
    doubly_linked_list<int> l;
    l.push_back(10);
    l.push_back(11);
    l.push_back(12);
    l.push_back(13);
    l.reverse();
    std::vector<int> els = l.elements();
    std::vector<int> v = {13, 12, 11, 10};
    REQUIRE(els == v);
}

TEST_CASE("testing iterators in doubly list") {
    doubly_linked_list<int> l({1, 5, 6, 3, 2, 4});
    std::vector<int> v = l.elements();
    std::vector<int> check;
    for (auto it = l.begin(); it != l.end(); it++) {
        check.push_back(*(it));
    }
    REQUIRE(v == check);

    auto it = l.begin();
    std::vector<int> v2;
    v2.push_back(*(it));
    it++;
    v2.push_back(*(it));
    it--;
    v2.push_back(*(it));
    std::vector<int> check2 = {1, 5, 1};
    REQUIRE(v2 == check2);
}

TEST_CASE("testing copy constructors in doubly list") {
    doubly_linked_list<int> l({1, 5, 12, 7, 8, 4});
    doubly_linked_list<int> l2(l);
    std::vector<int> v = l.elements();
    std::vector<int> v2 = l2.elements();
    REQUIRE(v == v2);
}

TEST_CASE("testing operator = in doubly list") {
    doubly_linked_list<int> l({1, 5, 12, 7, 8, 4});
    std::vector<int> v = l.elements();
    doubly_linked_list<int> l2;
    l2 = l;
    std::vector<int> v2 = l2.elements();
    REQUIRE(v2 == v);
}

TEST_CASE("Testing searching for issue #88") {
    // Bug in search method
    doubly_linked_list<int> l({1, 2, 3, 4});
    REQUIRE(l.search(4) == true);
    doubly_linked_list<char> ll({'a', 'b', 'c', 'd', 'e'});
    REQUIRE(ll.search('e') == true);
    ll.erase('e');
    REQUIRE(ll.search('e') == false);
    REQUIRE(ll.search('d') == true);
    ll.erase('d');
    REQUIRE(ll.search('d') == false);

    // Bug in push_front method
    doubly_linked_list<int> l2;
    l2.push_front(1);
    l2.push_back(2);
    l2.push_back(3);
    std::vector<int> check = {1, 2, 3};
    int idx = 0;
    for (auto it = l2.begin(); it != l2.end(); it++) {
        REQUIRE(*(it) == check[idx++]);
    }

    // Bug in erase method
    doubly_linked_list<int> l3({1, 2, 3});
    l3.erase(3);
    l3.push_back(4);
    l3.push_back(5);
    check.clear();
    check = {1, 2, 4, 5};
    idx = 0;
    for (auto it = l3.begin(); it != l3.end(); it++) {
        REQUIRE(*(it) == check[idx++]);
    }

    doubly_linked_list<int> l4({1, 2, 3});
    l4.erase(3);
    REQUIRE(l4.search(3) == false);
    l4.push_back(3);
    REQUIRE(l4.search(3) == true);
    l4.erase(3);
    l4.erase(2);
    REQUIRE(l4.search(2) == false);
    l4.push_back(2);
    REQUIRE(l4.search(2) == true);
    l4.erase(2);
    l4.push_front(2);
    REQUIRE(l4.search(2) == true);

    doubly_linked_list<int> l5({1, 2, 3});
    l5.erase(1);
    REQUIRE(l5.search(1) == false);
    l5.erase(3);
    REQUIRE(l5.search(3) == false);
}

TEST_CASE("Testing visualization of doubly linked list") {
    doubly_linked_list<int> l({1, 5, 12, 7, 8, 4});
    CHECK_NOTHROW(l.visualize());
}
