#include "../../../src/algorithms/searching/dfs.h"
#include "../../../third_party/catch.hpp"

TEST_CASE("testing depth first search") {

    std::unordered_map<int, std::vector<int>> adj;
    adj[1] = {1, 2, 3};
    adj[2] = {2, 3, 4};
    adj[3] = {3, 4, 5};

    std::unordered_set<int> visited;
    REQUIRE(dfs(adj, 1, 3) == true);
}

TEST_CASE("testing depth first search 2") {
    std::unordered_map<char, std::vector<char>> adj;
    adj['a'] = {'b', 'c'};
    adj['b'] = {'d', 'e'};
    adj['d'] = {'e', 'b'};
    adj['e'] = {'b', 'd'};
    adj['f'] = {'g'};
    adj['g'] = {'f'};
    REQUIRE(dfs(adj, 'a', 'f') == false);
}
