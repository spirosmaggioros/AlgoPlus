#include "../../../src/algorithms/searching/shortest_path.h"
#include "../../../third_party/catch.hpp"

TEST_CASE("Testing dijkstra's algorithm [1]") {
    std::unordered_map<int, std::vector<std::pair<int, int>>> adj;
    adj[0].push_back({1, 10});
    adj[2].push_back({3, 8});
    adj[1].push_back({2, 6});
    adj[0].push_back({3, 4});

    REQUIRE(shortest_path(adj, 4, 0, 3) == 4);
}

TEST_CASE("Testing dijkstra's algorithm [2]") {

    std::unordered_map<int, std::vector<std::pair<int, int>>> adj;

    adj[0].push_back({1, 0});
    adj[1].push_back({2, 2});
    adj[2].push_back({5, 3});
    adj[1].push_back({5, 1});
    adj[3].push_back({2, 4});
    REQUIRE(shortest_path(adj, 6, 0, 5) == 1);
}

TEST_CASE("Testing dijkstra's algorithm [3]") {
    std::unordered_map<int, std::vector<std::pair<int, int>>> adj;

    adj[0].push_back({0, 0});
    REQUIRE(shortest_path(adj, 5, 0, 4) == -1);

    adj[1].push_back({2, 1});
    adj[2].push_back({3, 4});
    adj[3].push_back({4, 5});
    adj[1].push_back({4, 6});
    REQUIRE(shortest_path(adj, 5, 1, 4) == 6);
}
