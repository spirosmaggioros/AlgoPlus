#include "../../src/helpers/timer.h"
#include "../../third_party/catch.hpp"
#include "../../src/classes/graph/graph.h"

TEST_CASE("Testing execution time for timer namespcae") {
    weighted_graph<int> g("directed");
    g.add_edge(0, 1, 5);
    g.add_edge(1, 2, 4);
    g.add_edge(2, 3, 5);
    g.add_edge(3, 4, 2);
    g.add_edge(4, 5, 1);

    // in order to pass member function, you need to make them lambda first,
    // if they're static function, this step could be ignored
    auto shortest_path = [&](int a, int b) -> double { return g.shortest_path(a, b); };

    auto dfs = [&](int a) -> std::vector<int> { return g.dfs(a); };

    CHECK_NOTHROW(TIMER::exec_time(shortest_path, 0, 4));
}

TEST_CASE("Testing time complxity for timer namespace") {
    weighted_graph<int> g("directed");
    g.add_edge(0, 1, 5);
    g.add_edge(1, 2, 4);
    g.add_edge(2, 3, 5);
    g.add_edge(3, 4, 2);
    g.add_edge(4, 5, 1);

    // in order to pass member function, you need to make them lambda first,
    // if they're static function, this step could be ignored
    auto shortest_path = [&](int a, int b) -> double { return g.shortest_path(a, b); };

    auto dfs = [&](int a) -> std::vector<int> { return g.dfs(a); };

    CHECK_NOTHROW(
        TIMER::time_complexity(dfs, std::make_tuple(0), std::make_tuple(2), std::make_tuple(4)));
}

TEST_CASE("Testing check faster for timer namespace") {
    weighted_graph<int> g("directed");
    g.add_edge(0, 1, 5);
    g.add_edge(1, 2, 4);
    g.add_edge(2, 3, 5);
    g.add_edge(3, 4, 2);
    g.add_edge(4, 5, 1);

    // in order to pass member function, you need to make them lambda first,
    // if they're static function, this step could be ignored
    auto shortest_path = [&](int a, int b) -> double { return g.shortest_path(a, b); };

    auto dfs = [&](int a) -> std::vector<int> { return g.dfs(a); };

    CHECK_NOTHROW(
        TIMER::check_faster(shortest_path, std::make_tuple(0, 1), dfs, std::make_tuple(0)));
}
