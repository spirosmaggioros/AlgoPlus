#define ENABLE_LIST_VISUALIZATION
#include "../../src/classes/list/frequency_list.h"
#include "../../third_party/catch.hpp"
#include <string>
#include <vector>

TEST_CASE("Testing empty() and push_front() frequency_list") {
    frequency_list<int> flist;

    REQUIRE(flist.empty() == true);
    flist.push_front(5);
    REQUIRE(flist.empty() == false);
}

TEST_CASE("Testing push_back() frequency_list") {
    frequency_list<int> flist;

    flist.push_back(7);
    REQUIRE(flist.get_frequency(7) == 1);
}

TEST_CASE("Testing search() frequency_list") {
    frequency_list<int> flist;

    flist.push_front(7);
    flist.push_front(10);
    flist.push_front(12);
    flist.push_front(13);
    REQUIRE(flist.search(7) == true);
    REQUIRE(flist.search(8) == false);
}

TEST_CASE("Testing erase() frequency_list") {
    frequency_list<int> flist;

    flist.push_back(10);
    flist.push_back(11);
    flist.erase(10);
    REQUIRE(flist.search(10) == false);
}

TEST_CASE("Testing functionality in unison frequency_list") {
    frequency_list<int> flist;

    flist.push_back(1);
    flist.push_back(1);
    flist.push_front(2);
    flist.push_front(2);
    flist.push_front(2);

    REQUIRE(flist.get_frequency(1) == 2);
    REQUIRE(flist.get_frequency(2) == 3);

    flist.erase(1);
    REQUIRE(flist.search(1) == false);
    REQUIRE(flist.search(2) == true);
}

TEST_CASE("Testing vector initialization for the frequency list class") {
    frequency_list<char> flist({'a', 'b', 'c', 'd', 'e', 'f'});
    std::vector<char> check;
    std::vector<char> v = {'a', 'b', 'c', 'd', 'e', 'f'};
    for (auto it = flist.begin(); it != flist.end(); it++) {
        check.push_back(*(it));
    }
    REQUIRE(check == v);

    frequency_list<std::string> f2({"hello", "world", "mars", "sun"});
    frequency_list<std::string> f3 = f2;

    std::vector<std::string> v1, v2;
    for (auto it = f2.begin(); it != f2.end(); it++) {
        v1.push_back(*(it));
    }
    for (auto it = f3.begin(); it != f3.end(); it++) {
        v2.push_back(*(it));
    }
    REQUIRE(v2 == v1);
}

TEST_CASE("Testing frequency list visualization") {
    frequency_list<char> flist({'a', 'b', 'c', 'd', 'e', 'f'});
    CHECK_NOTHROW(flist.visualize());
}
