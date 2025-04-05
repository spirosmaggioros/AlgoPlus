#include "../../src/classes/tree/trie.h"
#include "../../third_party/catch.hpp"
#include <string>

TEST_CASE("testing insertion in trie") {
    trie t;
    t.insert("hello");
    t.insert("world");
    REQUIRE(t.search("hello") == true);
    REQUIRE(t.search("helo") == false);
}

TEST_CASE("testing remove in trie") {
    trie t;
    t.insert("hello");
    t.insert("world");
    t.insert("mars");
    t.insert("universe");
    REQUIRE(t.search("universe") == true);
    t.remove("universe");
    REQUIRE(t.search("universe") == false);
}

TEST_CASE("testing copy constructor for trie class") {
    trie t({"abc", "bca", "dfs", "bfs"});
    trie t2(t);
    std::vector<std::string> v = {"abc", "bca", "dfs", "bfs"};
    for (auto& x : v) {
        REQUIRE(t2.search(x) == true);
    }
}

TEST_CASE("testing operator = for trie class") {
    trie t({"abc", "bca", "dfs", "bfs"});
    trie t2;
    t2 = t;
    std::vector<std::string> v = {"abc", "bca", "dfs", "bfs"};
    for (auto& x : v) {
        REQUIRE(t2.search(x) == true);
    }
}