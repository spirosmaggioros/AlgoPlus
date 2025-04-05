#include "../../../src/machine_learning/search_algorithms/hill_climbing/hill_climbing.h"
#include "../../../third_party/catch.hpp"
#include <string>

TEST_CASE("testing edges with insertion for hill climbing class") {
    hill_climbing<int> h;
    h.insert_node(0, INT_MAX);
    h.insert_node(1, 10);
    h.insert_node(2, 20);
    h.insert_node(3, 10);
    h.add_edge(0, 1);
    REQUIRE(h.has_edge(0, 1) == true);
    REQUIRE(h.has_edge(0, 3) == false);
}

TEST_CASE("Testing default constructor of hill climbing") {
    std::unordered_map<int, double> node;
    std::unordered_map<int, std::vector<std::pair<int, double>>> adj;

    node[0] = INT_MAX;
    node[1] = 10;
    node[2] = 20;
    node[3] = 10;

    adj[0].push_back({1, node[1]});

    hill_climbing<int> h(adj, node);

    REQUIRE(h.has_edge(0, 1) == true);
    REQUIRE(h.has_edge(0, 3) == false);
}

TEST_CASE("testing search function for hill climbing class") {
    hill_climbing<char> h;
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

    hill_climbing<char> h2;
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
