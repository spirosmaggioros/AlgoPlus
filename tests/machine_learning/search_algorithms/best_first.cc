#include "../../../src/machine_learning/search_algorithms/best_first/best_first.h"
#include "../../../third_party/catch.hpp"
#include <string>

TEST_CASE("testing edges with insertion for best first class") {
    best_first<int> h;
    h.insert_node(0, INT_MAX);
    h.insert_node(1, 10);
    h.insert_node(2, 20);
    h.insert_node(3, 10);
    h.add_edge(0, 1);
    REQUIRE(h.has_edge(0, 1) == true);
    REQUIRE(h.has_edge(0, 3) == false);
}

TEST_CASE("Testing default constructor of best first with passed adjacency list") {
    std::unordered_map<int, std::vector<std::pair<int, double>>> adj;
    std::unordered_map<int, double> nodes;

    nodes[0] = INT_MAX;
    nodes[1] = 10;
    nodes[2] = 20;
    nodes[3] = 10;

    adj[0].push_back({1, nodes[1]});

    best_first<int> h(adj, nodes);
    REQUIRE(h.has_edge(0, 1) == true);
    REQUIRE(h.has_edge(0, 3) == false);
    REQUIRE(h.has_edge(1, 0) == false);
}

TEST_CASE("testing search function for best_first class") {
    best_first<char> h;
    h.insert_node('s', INT_MAX);
    h.insert_node('A', 10.4);
    h.insert_node('B', 6.7);
    h.insert_node('C', 3.2);
    h.insert_node('D', 5.6);
    h.insert_node('E', 5.3);
    h.insert_node('F', 4.33);
    h.insert_node('G', 0.0);
    h.add_edge('s', 'D');
    h.add_edge('A', 'B');
    h.add_edge('A', 'D');
    h.add_edge('D', 'E');
    h.add_edge('E', 'F');
    h.add_edge('F', 'G');
    REQUIRE(h.search('s', 'G') == true);

    best_first<char> h2;
    h2.insert_node('s', INT_MAX);
    h2.insert_node('A', 10.4);
    h2.insert_node('B', 6.7);
    h2.insert_node('C', 3.2);
    h2.insert_node('D', 5.6);
    h2.insert_node('E', 5.9);
    h2.insert_node('F', 4.33);
    h2.insert_node('G', 0.0);
    h2.add_edge('s', 'D');
    h2.add_edge('A', 'B');
    h2.add_edge('A', 'D');
    h2.add_edge('D', 'E');
    h2.add_edge('E', 'F');
    h2.add_edge('F', 'G');

    REQUIRE(h2.search('s', 'G') == false);
}

TEST_CASE("Testing throws on add_edge for best first function") {
    std::unordered_map<int, std::vector<std::pair<int, double>>> adj;
    std::unordered_map<int, double> nodes;

    nodes[0] = INT_MAX;
    nodes[1] = 10;
    nodes[2] = 20;
    nodes[3] = 10;

    adj[0].push_back({1, nodes[1]});

    best_first<int> h(adj, nodes);

    CHECK_NOTHROW(h.add_edge(10, 20));
    CHECK_NOTHROW(h.add_edge(0, 1));
}
