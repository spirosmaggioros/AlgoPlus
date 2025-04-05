#ifndef GRAPH_H
#include "../../src/classes/graph/graph.h"
#endif

int main() {
    weighted_graph<char> graph("directed");
    graph.add_edge('a', 'b', 1);
    graph.add_edge('b', 'c', 1);
    graph.add_edge('b', 'f', 4);
    graph.add_edge('b', 'g', 4);
    graph.add_edge('c', 'e', 1);
    graph.add_edge('d', 'c', 1);
    graph.add_edge('e', 'd', -3);
    graph.add_edge('f', 'g', 5);
    graph.add_edge('f', 'h', 3);
    graph.add_edge('g', 'h', 4);

    char start = 'a';
    std::unordered_map<char, double> dist = graph.bellman_ford(start);

    for (auto it : dist)
        printf("The cost to get from node %c to %c is %.2f\n", start, it.first, it.second);

    // Output:
    // The cost to get from node a to h is 0.00
    // The cost to get from node a to a is 0.00
    // The cost to get from node a to b is 1.00
    // The cost to get from node a to c is -inf
    // The cost to get from node a to d is -inf
    // The cost to get from node a to e is -inf
    // The cost to get from node a to f is 5.00
    // The cost to get from node a to g is 5.00

    return 0;
}
