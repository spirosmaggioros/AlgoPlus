#include "../../../src/algorithms/searching/bfs.h"
#include "../../../third_party/catch.hpp"

TEST_CASE("testing breadth first search") {
    std::unordered_map<int, std::vector<int>> adj;
    adj[1] = {1, 3, 4};
    adj[2] = {1, 4};

    REQUIRE(bfs(adj, 1, 4) == true);
}

TEST_CASE("testing breadth first search 2") {
    std::unordered_map<char, std::vector<char>> adj;
    adj['a'] = {'b', 'c'};
    adj['b'] = {'d', 'e'};
    adj['d'] = {'e', 'b'};
    adj['e'] = {'b', 'd'};
    adj['f'] = {'g'};
    adj['g'] = {'f'};
    REQUIRE(bfs(adj, 'a', 'f') == false);
}
