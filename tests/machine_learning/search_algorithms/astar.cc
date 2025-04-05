#include "../../../src/machine_learning/search_algorithms/AStar/astar.h"
#include "../../../third_party/catch.hpp"
#include <string>

TEST_CASE("Testing edge creation in A*") {
    AStar<int> adj;
    adj.insert_node(1, 10);
    adj.insert_node(2, 5);
    adj.insert_node(3, 5);
    adj.insert_node(10, 10);
    adj.add_edge(1, 3, 5);
    adj.add_edge(1, 2, 7);
    adj.add_edge(3, 10, 5);

    REQUIRE(adj.has_edge(1, 3) == true);
    REQUIRE(adj.has_edge(1, 2) == true);
    REQUIRE(adj.has_edge(2, 1) == false);
    REQUIRE(adj.has_edge(3, 2) == false);
}

TEST_CASE("Testing shortest path with admissible heuristic") {
    AStar<std::string> a;
    a.insert_node("S", 6);
    a.insert_node("A", 4);
    a.insert_node("B", 4);
    a.insert_node("D", 2);
    a.insert_node("C", 2);
    a.insert_node("G", 0);

    a.add_edge("S", "A", 3.0);
    a.add_edge("S", "B", 3.0);
    a.add_edge("A", "B", 1.0);
    a.add_edge("A", "D", 5.0);
    a.add_edge("B", "C", 2.0);
    a.add_edge("B", "D", 3.0);
    a.add_edge("C", "G", 2.0);
    a.add_edge("C", "D", 3.0);

    std::vector<std::string> path = a.shortest_path("S", "G");
    std::vector<std::string> check = {"S", "B", "C", "G"};
    REQUIRE(path == check);
}

TEST_CASE("Testing shortest path with non admissible heuristic") {
    AStar<std::string> a;
    a.insert_node("A", 8);
    a.insert_node("B", 7);
    a.insert_node("C", 0);
    a.insert_node("D", 0);

    a.add_edge("A", "B", 1);
    a.add_edge("B", "A", 1);
    a.add_edge("B", "C", 1);
    a.add_edge("C", "B", 1);
    a.add_edge("C", "D", 6);
    a.add_edge("D", "C", 6);
    a.add_edge("A", "C", 3);
    a.add_edge("C", "A", 3);

    std::vector<std::string> path = a.shortest_path("A", "D");
    std::vector<std::string> check = {"A", "B", "C", "D"};
    REQUIRE(path == check);
}
