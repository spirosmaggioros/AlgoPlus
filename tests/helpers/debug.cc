#include "../../src/helpers/debug.h"
#include "../../third_party/catch.hpp"
#include "../../src/classes/graph/graph.h"

TEST_CASE("Testing get args for debug namespace") {
    weighted_graph<int> g("directed");
    g.add_edge(0, 1, 5);
    g.add_edge(1, 2, 4);
    g.add_edge(2, 3, 5);
    g.add_edge(3, 4, 2);
    g.add_edge(4, 5, 1);

    // in order to pass member function, you need to make them lambda first,
    // if they're static function, this step could be ignored
    auto shortest_path = [&](int a, int b) -> double { return g.shortest_path(a, b); };

    CHECK_NOTHROW(DEBUG::get_args(shortest_path, 0, 2));
}

TEST_CASE("Testing get args for debug namespace with vectors") {
    auto it = [&](int a, int b) -> std::vector<int> { return {a, b}; };
    std::vector<int> passed = {2, 4};
    CHECK_NOTHROW(DEBUG::get_args(it, passed));
}
