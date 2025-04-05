#ifndef GRAPH_H
#include "../../src/classes/graph/graph.h"
#endif

#ifdef __cplusplus
#include <iostream>
#include <string>
#endif

int main() {
    weighted_graph<std::string> g("directed");
    g.add_edge("Athens", "Benfica", 300);
    g.add_edge("Athens", "Milan", 120);
    g.add_edge("Milan", "Paris", 40);
    g.add_edge("Paris", "Benfica", 120);
    g.add_edge("Rome", "Malta", 110);
    g.add_edge("Milan", "Malta", 250);
    g.add_edge("Malta", "Berlin", 190);
    g.add_edge("Athens", "Rome", 70);

    std::vector<std::string> topo = g.topological_sort();

    std::cout << g.shortest_path("Athens", "Malta") << '\n';
    std::cout << g.shortest_path("Athens", "Benfica") << '\n';
}
