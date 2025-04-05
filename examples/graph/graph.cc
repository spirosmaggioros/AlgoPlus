#include "../../src/classes/graph/graph.h"
#include "../../src/visualization/graph_visual/graph_visualization.h"

#ifdef __cplusplus
#include <iostream>
#endif

int main() {
    weighted_graph<int> g("undirected");
    g.add_edge(1, 3, 4);
    g.add_edge(3, 4, 5);
    g.add_edge(0, 1, 6);
    g.add_edge(0, 2, 7);

    std::cout << g.shortest_path(0, 10) << '\n';

    graph<int> g2("directed", {{1, {2, 3}}, {2, {4, 5}}});
    std::vector<int> v = g2.bfs(0);
    for (auto& x : v) {
        std::cout << x << " ";
    }
    std::cout << '\n';

    // now you can visualize graphs with algoplus
    g.visualize();
}
