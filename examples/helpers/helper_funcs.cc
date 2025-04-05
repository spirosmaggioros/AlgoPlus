#include <iostream>
#include "../../src/algorithms/number_theory/eratosthenes_sieve.h"
#include "../../src/classes/graph/graph.h"
#include "../../src/helpers/debug.h"
#include "../../src/helpers/timer.h"

int main() {
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

    // you can even do this: auto bound_dfs = std::bind(&graph<int>::dfs, &g,
    // std::placeholders::_1);
    TIMER::exec_time(shortest_path, 0, 4);

    // printing all the arguments(0, 2, 3, 4, 'a')
    DEBUG::get_args(shortest_path, 0, 2, 3, 4, 'a');

    // returns which function is faster
    std::cout << TIMER::check_faster(shortest_path, std::make_tuple(0, 1), dfs, std::make_tuple(0))
              << '\n';

    // analyzes the time complexity, also returns a useful graph
    TIMER::time_complexity(soe, std::make_tuple(4), std::make_tuple(5), std::make_tuple(10),
                           std::make_tuple(15), std::make_tuple(50), std::make_tuple(100),
                           std::make_tuple(500), std::make_tuple(10000), std::make_tuple(99999));
}
