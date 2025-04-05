#include "../../../src/algorithms/geometry/convex_hull.h"
#include "../../../third_party/catch.hpp"

TEST_CASE("Testing convex hull with graham's scan [1]") {
    std::vector<std::pair<int, int>> points = {{0, 3}, {1, 1}, {2, 2}, {4, 4}, {0, 0},
                                               {1, 2}, {3, 1}, {3, 3}, {2, 4}, {4, 0}};
    std::vector<std::pair<int, int>> points2 = points;

    std::vector<std::pair<int, int>> convex = graham_scan(points);
    std::vector<std::pair<int, int>> check = {{0, 0}, {4, 0}, {4, 4}, {2, 4}, {0, 3}, {0, 0}};
    REQUIRE(convex == check);

    std::vector<std::pair<int, int>> convex2 = graham_scan(points2, false);
    check.pop_back();
    REQUIRE(convex2 == check);
}

TEST_CASE("Testing convex hull with graham's scan [2]") {
    std::vector<std::pair<int, int>> points = {{1, 1}, {2, 2}, {0, 3}, {3, 0},
                                               {0, 0}, {3, 3}, {3, 1}, {1, 3}};
    std::vector<std::pair<int, int>> points2 = points;

    std::vector<std::pair<int, int>> convex = graham_scan(points);
    std::vector<std::pair<int, int>> check = {{0, 0}, {3, 0}, {3, 3}, {0, 3}, {0, 0}};

    REQUIRE(convex == check);

    std::vector<std::pair<int, int>> convex2 = graham_scan(points2, false);
    check.pop_back();
    REQUIRE(convex2 == check);
}
