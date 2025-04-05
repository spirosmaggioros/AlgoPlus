#include "../../src/classes/hash_table/hash_table.h"
#include "../../third_party/catch.hpp"
#include <list>
#include <string>

TEST_CASE("checking insertions, overriding insertions and retrievals in hash table") {
    hash_table<std::string, int> table;

    table.insert("abc", 1);
    table.insert("abc", 2);
    table.insert("def", 3);

    REQUIRE(table.retrieve("abc") == 2);
    REQUIRE(table.retrieve("ghi") == std::nullopt);
}

TEST_CASE("checking removals in hash table") {
    hash_table<std::string, int> table;

    table.insert("abc", 1);
    table.insert("def", 2);
    table.remove("abc");

    REQUIRE(table.retrieve("abc") == std::nullopt);
    REQUIRE(table.retrieve("def") == 2);
}

TEST_CASE("testing copy constructor in hash table") {
    hash_table<int, std::string> table;
    table.insert(1, "abc");
    table.insert(2, "bcd");
    table.insert(3, "cba");

    hash_table<int, std::string> table2(table);

    for (int i = 1; i <= 3; i++) {
        REQUIRE(table.retrieve(i) == table2.retrieve(i));
    }
}

TEST_CASE("testing operator = in hash table") {
    hash_table<int, std::string> table;
    table.insert(1, "abc");
    table.insert(2, "bcd");
    table.insert(3, "cba");

    hash_table<int, std::string> table2 = table;

    for (int i = 1; i <= 3; i++) {
        REQUIRE(table.retrieve(i) == table2.retrieve(i));
    }
}

TEST_CASE("checking hash_table<int, std::string> table") {
    hash_table<int, std::string> table;
    table.insert(1, "abc");
    table.insert(2, "bcd");
    table.insert(3, "cba");

    std::vector<std::pair<int, std::string>> tablePairs;
    for (auto it = table.begin(); it != table.end(); it++) {
        tablePairs.push_back(*it);
    }
    std::sort(tablePairs.begin(), tablePairs.end());

    std::vector<std::pair<int, std::string>> check_table = {{1, "abc"}, {2, "bcd"}, {3, "cba"}};
    std::sort(check_table.begin(), check_table.end());

    REQUIRE(tablePairs == check_table);
}

TEST_CASE("checking hash_table<int, std::string> t2") {
    hash_table<int, std::string> t2;
    t2.insert(1, "a");
    t2.insert(2, "b");
    t2.insert(3, "c");
    t2.insert(4, "d");

    std::vector<std::pair<int, std::string>> t2Pairs;
    for (auto it = t2.begin(); it != t2.end(); it++) {
        t2Pairs.push_back(*it);
    }
    std::sort(t2Pairs.begin(), t2Pairs.end());

    std::vector<std::pair<int, std::string>> check_t2 = {{1, "a"}, {2, "b"}, {3, "c"}, {4, "d"}};
    std::sort(check_t2.begin(), check_t2.end());

    REQUIRE(t2Pairs == check_t2);
}

TEST_CASE("checking hash_table<int, std::string> t3") {
    hash_table<int, std::string> t3;
    t3.insert(1, "a");
    t3.insert(2, "b");
    t3.insert(3, "c");
    t3.insert(4, "d");

    std::vector<std::pair<int, std::string>> check_vec = {{1, "a"}, {2, "b"}, {3, "c"}, {4, "d"}};
    std::sort(check_vec.begin(), check_vec.end());

    std::vector<std::pair<int, std::string>> vec;
    for (auto it = t3.begin(); it != t3.end(); it++) {
        vec.push_back(*it);
    }
    std::sort(vec.begin(), vec.end());

    REQUIRE(vec == check_vec);
}

TEST_CASE("Testing operator = for hash table class") {
    hash_table<int, std::string> t3;
    t3.insert(1, "a");
    t3.insert(2, "b");
    t3.insert(3, "c");
    t3.insert(4, "d");

    std::vector<std::pair<int, std::string>> check_vec = {{1, "a"}, {2, "b"}, {3, "c"}, {4, "d"}};
    hash_table<int, std::string> t = t3;
    std::vector<std::pair<int, std::string>> vec;
    for (auto it = t3.begin(); it != t3.end(); it++) {
        vec.push_back(*it);
    }
    std::sort(vec.begin(), vec.end());
    REQUIRE(check_vec == vec);
}

TEST_CASE("Testing operator << for hash table class") {
    hash_table<int, std::string> t3;
    t3.insert(1, "a");
    t3.insert(2, "b");
    t3.insert(3, "c");
    t3.insert(4, "d");

    CHECK_NOTHROW(std::cout << t3 << '\n');
}

TEST_CASE("Testing iterator -- for hash table class") {
    hash_table<int, std::string> t3;
    t3.insert(1, "a");
    t3.insert(2, "b");
    t3.insert(3, "c");
    t3.insert(4, "d");

    auto it = t3.end();
    CHECK_NOTHROW(it--);
    CHECK_NOTHROW(it--);
    CHECK_NOTHROW(it--);
}
