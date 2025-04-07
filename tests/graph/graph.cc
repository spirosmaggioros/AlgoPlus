#define ENABLE_GRAPH_VISUALIZATION
#include "../../src/classes/graph/graph.h"
#include "../../third_party/catch.hpp"
#include <string>

TEST_CASE("testing clearing of a graph") {
    graph<int> g("directed", {{0, {1, 2}}, {1, {3, 4}}, {2, {4, 5}}});
    g.clear();
    std::vector<int> v = g.topological_sort();
    std::vector<int> v2 = g.dfs(0);
    REQUIRE(v2.empty() == true);
    REQUIRE(v.empty() == true);
}

TEST_CASE("testing dfs in graph") {
    graph<int> g("directed");
    g.add_edge(1, 3);
    g.add_edge(3, 4);
    g.add_edge(4, 5);

    std::vector<int> v = {1, 3, 4, 5};
    std::vector<int> dfs = g.dfs(1);
    REQUIRE(v == dfs);
}

TEST_CASE("testing dfs in graph [2]") {
    graph<int> g("directed");
    g.add_edge(1, 4);
    g.add_edge(4, 5);
    g.add_edge(6, 7);
    g.add_edge(7, 8);
    std::vector<int> v1 = {1, 4, 5}, v2 = {6, 7, 8};
    std::vector<int> dfs1 = g.dfs(1), dfs2 = g.dfs(6);
    REQUIRE(v1 == dfs1);
    REQUIRE(v2 == dfs2);
}

TEST_CASE("testing connecting components in graph") {
    graph<int> g("undirected");
    g.add_edge(1, 2);
    g.add_edge(2, 4);
    g.add_edge(5, 6);
    g.add_edge(7, 8);
    REQUIRE(g.connected_components() == 3);

    graph<char> g2("undirected");
    g2.add_edge('a', 'g');
    g2.add_edge('b', 'o');
    g2.add_edge('o', 'a');
    g2.add_edge('w', 'e');
    REQUIRE(g2.connected_components() == 2);
}

TEST_CASE("testing cycle detection in graph") {
    graph<int> g("undirected");
    g.add_edge(1, 2);
    g.add_edge(2, 3);
    g.add_edge(3, 1);
    REQUIRE(g.cycle() == true);

    graph<int> g2("directed");
    g2.add_edge(1, 2);
    g2.add_edge(2, 3);
    g2.add_edge(3, 1);
    REQUIRE(g2.cycle() == true);

    graph<int> g3("directed");
    g3.add_edge(4, 5);
    g3.add_edge(5, 6);
    g3.add_edge(4, 6);
    REQUIRE(g3.cycle() == false);

    graph<char> g4("directed");
    g4.add_edge('a', 'c');
    g4.add_edge('c', 'a');
    REQUIRE(g4.cycle() == true);
}

TEST_CASE("testing topological sorting in graph") {
    graph<int> g("directed");
    g.add_edge(0, 1);
    g.add_edge(1, 3);
    g.add_edge(2, 3);
    g.add_edge(3, 4);
    std::vector<int> v1 = {2, 0, 1, 3, 4};
    REQUIRE(g.topological_sort() == v1);
}

TEST_CASE("testing bipartite check in graph") {
    graph<char> g("undirected");
    g.add_edge('a', 'b');
    g.add_edge('b', 'c');
    g.add_edge('c', 'd');
    g.add_edge('d', 'a');
    REQUIRE(g.bipartite() == true);
}

TEST_CASE("testing size in graph") {
    graph<std::string> g("directed");
    g.add_edge("hello", "there");
    g.add_edge("i'm", "batman");
    g.add_edge("batman", "hero");
    g.add_edge("hero", "hello");
    REQUIRE(g.size() == 5);
}

TEST_CASE("testing if an edge exists in graph") {
    graph<int> g("directed");
    g.add_edge(1, 5);
    g.add_edge(5, 10);
    g.add_edge(3, 5);
    g.add_edge(5, 1);
    REQUIRE(g.has_edge(1, 5) == true);
    REQUIRE(g.has_edge(1, 10) == false);
    REQUIRE(g.has_edge(5, 1) == true);

    graph<int> g2("undirected");
    g2.add_edge(1, 3);
    g2.add_edge(5, 10);
    g2.add_edge(3, 5);
    g2.add_edge(1, 10000);
    REQUIRE(g2.has_edge(1, 10000) == true);
    REQUIRE(g2.has_edge(10, 5) == true);
    REQUIRE(g2.has_edge(1, 5) == false);
}

TEST_CASE("testing bridge detection in graph") {
    graph<int> g("undirected");
    g.add_edge(1, 0);
    g.add_edge(0, 2);
    g.add_edge(1, 2);
    g.add_edge(0, 3);
    g.add_edge(3, 4);

    std::vector<std::vector<int>> bridges = {{4, 3}, {3, 0}};
    REQUIRE(g.bridge(0) == bridges);
}

TEST_CASE("testing connectivity in graph") {
    graph<int> g("undirected");
    g.add_edge(1, 0);
    g.add_edge(0, 2);
    g.add_edge(1, 2);
    g.add_edge(0, 3);
    g.add_edge(3, 4);
    g.add_edge(4, 0);
    REQUIRE(g.connected() == true);

    graph<int> g2("undirected");
    g2.add_edge(0, 1);
    g2.add_edge(5, 6);
    REQUIRE(g2.connected() == false);
}

TEST_CASE("testing eulerian check in graph") {
    graph<int> g("undirected");
    g.add_edge(1, 3);
    g.add_edge(1, 0);
    g.add_edge(1, 2);
    g.add_edge(2, 0);
    g.add_edge(0, 3);
    g.add_edge(3, 4);

    REQUIRE(g.eulerian() == 0);

    graph<int> g2("undirected");
    g2.add_edge(1, 2);
    g2.add_edge(2, 0);
    g2.add_edge(1, 0);
    g2.add_edge(0, 3);
    g2.add_edge(3, 4);
    REQUIRE(g2.eulerian() == 1);

    graph<int> g3("undirected");
    g3.add_edge(1, 0);
    g3.add_edge(1, 2);
    g3.add_edge(2, 0);
    g3.add_edge(0, 3);
    g3.add_edge(0, 4);
    g3.add_edge(3, 4);
    REQUIRE(g3.eulerian() == 2);
}

TEST_CASE("testing copy constructor in graph") {
    graph<int> g("undirected");
    g.add_edge(1, 3);
    g.add_edge(1, 0);
    g.add_edge(1, 2);
    g.add_edge(2, 0);
    g.add_edge(0, 3);
    g.add_edge(3, 4);
    graph<int> g1(g);
    graph<int> g2 = g;
    REQUIRE(g.dfs(1) == g1.dfs(1));
    REQUIRE(g.dfs(1) == g2.dfs(1));

    graph<char> g3("directed");
    g3.add_edge('a', 'b');
    g3.add_edge('b', 'd');
    g3.add_edge('c', 'd');
    g3.add_edge('w', 'd');
    g3.add_edge('d', 'a');

    graph<char> g4 = g3;
    graph<char> g5(g3);
    REQUIRE(g4.dfs('a') == g3.dfs('a'));
    REQUIRE(g5.dfs('a') == g3.dfs('a'));
}

TEST_CASE("testing operator = for graph class") {
    graph<int> g("undirected");
    g.add_edge(0, 1);
    g.add_edge(1, 0);
    g.add_edge(2, 3);
    g.add_edge(3, 2);
    graph<int> g2 = g;
    REQUIRE(g2.has_edge(0, 1));
    REQUIRE(g.has_edge(0, 1));
    REQUIRE(g2.has_edge(1, 0));
    REQUIRE(g.has_edge(1, 0));
    REQUIRE(g2.has_edge(2, 3));
    REQUIRE(g.has_edge(2, 3));
    REQUIRE(g2.has_edge(3, 2));
    REQUIRE(g.has_edge(3, 2));
}

TEST_CASE("Testing operator << for graph class") {
    graph<int> g("directed");
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    g.add_edge(2, 3);
    g.add_edge(3, 4);
    CHECK_NOTHROW(std::cout << g << '\n');

    graph<char> g2("directed");
    g2.add_edge('a', 'b');
    g2.add_edge('b', 'c');
    CHECK_NOTHROW(std::cout << g2 << '\n');
}

TEST_CASE("Testing dfs function for graph class [3]") {
    graph<int> g("directed");
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    g.add_edge(2, 3);
    g.add_edge(3, 4);
    std::vector<int> dfs_path = g.dfs(0);
    std::vector<int> path = {0, 1, 2, 3, 4};
    REQUIRE(dfs_path == path);
}

TEST_CASE("Testing bfs function for graph class [1]") {
    graph<int> g("directed");
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    g.add_edge(2, 3);
    g.add_edge(3, 4);
    std::vector<int> bfs_path = g.bfs(0);
    std::vector<int> path = {0, 1, 2, 3, 4};
    REQUIRE(bfs_path == path);
}

TEST_CASE("Testing operator = for graph class") {
    graph<int> g("directed");
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    g.add_edge(2, 3);
    g.add_edge(3, 4);

    graph<int> g2 = g;
    std::vector<int> dfs_path = g.dfs(0);
    std::vector<int> dfs_path_2 = g2.dfs(0);
    REQUIRE(dfs_path == dfs_path_2);
}

TEST_CASE("Testing clear function for graph class") {
    graph<int> g("directed");
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    g.add_edge(2, 3);
    g.add_edge(3, 4);

    g.clear();
    std::vector<int> bfs_path = g.bfs(0);
    REQUIRE(bfs_path.size() == 0);
}

TEST_CASE("Testing scc function for graph class") {
    graph<int> g("directed");
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    g.add_edge(2, 0);
    g.add_edge(1, 3);
    g.add_edge(3, 4);
    g.add_edge(4, 3);

    REQUIRE(g.scc() == 2);

    graph<int> g2("undirected");
    REQUIRE(g2.scc() == 0);

    g2.add_edge(0, 1);
    g2.add_edge(1, 2);
    g2.add_edge(2, 0);
    g2.add_edge(3, 4);
    g2.add_edge(4, 5);
    g2.add_edge(5, 3);
    g2.add_edge(3, 6);
    g2.add_edge(1, 3);

    REQUIRE(g2.scc() == 1);

    graph<char> t("directed");
    t.add_edge('a', 'b');
    t.add_edge('b', 'c');
    t.add_edge('c', 'a');
    t.add_edge('b', 'd');
    t.add_edge('d', 'e');
    t.add_edge('e', 'd');

    REQUIRE(t.scc() == 2);

    graph<int> g3("directed");
    g3.add_edge(0, 1);
    g3.add_edge(1, 2);
    g3.add_edge(2, 0);
    g3.add_edge(1, 3);
    g3.add_edge(3, 4);
    g3.add_edge(4, 5);
    g3.add_edge(5, 3);
    g3.add_edge(3, 6);
    g3.add_edge(6, 7);
    g3.add_edge(6, 8);
    g3.add_edge(8, 6);

    REQUIRE(g3.scc() == 4);
}

TEST_CASE("Testing visualize for graph class") {
    graph<int> g("directed");
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    g.add_edge(2, 3);
    g.add_edge(3, 4);
    CHECK_NOTHROW(g.visualize());
}
