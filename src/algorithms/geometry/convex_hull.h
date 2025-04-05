#ifndef CONVEX_HULL_H
#define CONVEX_HULL_H

#include <cmath>
#ifdef __cplusplus
#include <algorithm>
#include <iostream>
#include <stack>
#include <vector>
#endif

namespace helpers {
/**
 * @brief orientation function
 * @param a: pair<int, int> the first point
 * @param b: pair<int, int> the second point
 * @param c: pair<int, int> the third point
 * @return -1: if it's clockwise
 * @return 1: if it's counter clockwise
 * @return 0: otherwise
 */
int orientation(const std::pair<int, int> a, const std::pair<int, int> b,
                const std::pair<int, int> c) {
    double v = a.first * (b.second - c.second) + b.first * (c.second - a.second) +
               c.first * (a.second - b.second);
    if (v < 0) {
        return -1;
    }
    if (v > 0) {
        return 1;
    }
    return 0;
}
}; // namespace helpers

std::vector<std::pair<int, int>> graham_scan(std::vector<std::pair<int, int>> points,
                                             bool include_first_twice = true) {
    // find the leftmost and downmost point
    std::pair<int, int> p0 = *min_element(
        points.begin(), points.end(), [](std::pair<int, int> a, std::pair<int, int> b) {
            return std::make_pair(a.second, a.first) < std::make_pair(b.second, b.first);
        });

    // sort the points by their polar angle(anti-clockwise)
    std::sort(points.begin(), points.end(),
              [&](const std::pair<int, int> a, const std::pair<int, int> b) {
                  int ori = helpers::orientation(p0, a, b);
                  if (ori == 0) {
                      return (p0.first - a.first) * (p0.first - a.first) +
                                 (p0.second - a.second) * (p0.second - a.second) <
                             (p0.first - b.first) * (p0.first - b.first) +
                                 (p0.second - b.second) * (p0.second - b.second);
                  }
                  return ori > 0;
              });

    std::vector<std::pair<int, int>> s;
    for (const auto& point : points) {
        while (s.size() > 1 && helpers::orientation(s[s.size() - 2], s.back(), point) <= 0) {
            s.pop_back();
        }
        s.push_back(point);
    }

    if (include_first_twice) {
        s.push_back(p0); // push back the starting point again
    }
    return s;
}

#endif
