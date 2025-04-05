#include "../containers/table.h"
#include "../../../third_party/catch.hpp"
#include <string>

TEST_CASE("Testing push_back function for table class") {
    table<int> t;
    t.push_back(10, 20, 30, 40, 50);
    std::vector<int> v = t.vectorize();
    std::vector<int> check = {10, 20, 30, 40, 50};
    REQUIRE(v == check);

    table<char> t2;
    t2.push_back('a', 'b', 'c', 'd', 'e');
    std::vector<char> v2 = t2.vectorize();
    std::vector<char> check2 = {'a', 'b', 'c', 'd', 'e'};
    REQUIRE(v2 == check2);

    table<std::string> t3;
    t3.push_back("a", "b", "c", "d", "e");
    std::vector<std::string> v3 = t3.vectorize();
    std::vector<std::string> check3 = {"a", "b", "c", "d", "e"};
    REQUIRE(v3 == check3);
}

TEST_CASE("Testing push_front function for table class") {
    table<int> t;
    t.push_front(50, 40, 30, 20, 10);
    std::vector<int> v = t.vectorize();
    std::vector<int> check = {10, 20, 30, 40, 50};
    REQUIRE(v == check);

    table<char> t2;
    t2.push_front('a', 'b', 'c', 'd', 'e');
    std::vector<char> v2 = t2.vectorize();
    std::vector<char> check2 = {'e', 'd', 'c', 'b', 'a'};
    REQUIRE(v2 == check2);

    table<std::string> t3;
    t3.push_front("a", "b", "c", "d", "e");
    std::vector<std::string> v3 = t3.vectorize();
    std::vector<std::string> check3 = {"e", "d", "c", "b", "a"};
    REQUIRE(v3 == check3);
}

TEST_CASE("Testing pop_front function for table class") {
    table<int> t;
    t.push_back(10, 20, 30, 40, 50);
    t.pop_front();
    REQUIRE(t[0] == 20);
    t.pop_front();
    REQUIRE(t[0] == 30);
    t.pop_front();
    REQUIRE(t[0] == 40);
    t.pop_front();
    REQUIRE(t[0] == 50);
    t.pop_front();
    REQUIRE(t.size() == 0);
    t.pop_front();
    REQUIRE(t.size() == 0);
}

TEST_CASE("Testing pop_back fucntion for table class") {
    table<int> t;
    t.push_back(10, 20, 30, 40, 50);
    t.pop_back();
    REQUIRE(t[t.size() - 1] == 40);
    t.pop_back();
    REQUIRE(t[t.size() - 1] == 30);
    t.pop_back();
    REQUIRE(t[t.size() - 1] == 20);
    t.pop_back();
    REQUIRE(t[t.size() - 1] == 10);
    t.pop_back();
    REQUIRE(t.size() == 0);
    t.pop_back();
    REQUIRE(t.size() == 0);
}

TEST_CASE("Testing iterators for table class") {
    table<int> t;
    t.push_back(10, 20, 30, 40, 50);
    std::vector<int> check = t.vectorize(), v;
    for (auto it = t.begin(); it != t.end(); it++) {
        v.push_back(*it);
    }
    REQUIRE(check == v);
}

TEST_CASE("Testing operator << for table class") {
    table<int> t;
    t.push_back(10, 20, 30, 40, 50);
    CHECK_NOTHROW(std::cout << t << '\n');

    table<char> tt;
    CHECK_NOTHROW(std::cout << tt << '\n');
}

TEST_CASE("Testing operator = for table class") {
    table<int> t;
    t.push_back(10, 20, 30, 40, 50);
    table<int> tt = t;
    std::vector<int> v1 = t.vectorize(), v2 = tt.vectorize();
    REQUIRE(v1 == v2);
    REQUIRE(t.size() == tt.size());
}

TEST_CASE("Testing copy constructor for table class") {
    table<int> t;
    t.push_back(10, 20, 30, 40, 50);
    table<int> tt(t);
    std::vector<int> v1 = t.vectorize(), v2 = tt.vectorize();
    REQUIRE(v1 == v2);
    REQUIRE(t.size() == tt.size());
}

TEST_CASE("Testing [] operator for table class") {
    table<int> t;
    t.push_back(10, 20, 30, 40, 50);
    std::vector<int> v{t.vectorize()};
    for (int i = 0; i < t.size(); i++) {
        REQUIRE(t[i] == v[i]);
    }

    REQUIRE(t[2] == 30);
    REQUIRE(t[4] == 50);
    REQUIRE(t[0] == 10);
}

TEST_CASE("Testing empty function for table class") {
    table<int> t;
    t.push_back(10, 20, 30, 40, 50);
    REQUIRE(t.empty() == false);
    int size = t.size();
    for (int i = 0; i < size; i++) {
        t.pop_back();
    }
    REQUIRE(t.empty() == true);
}
