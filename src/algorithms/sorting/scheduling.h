#ifndef SCHEDULING_H
#define SCHEDULING_H

#ifdef __cplusplus
#include <algorithm>
#include <iostream>
#include <vector>
#endif

/**
 * @brief scheduling function
 * @details Returns the maximum jobs that can be completed in a schedule with
 * pairs (x, y) indicating the starting and ending time
 * @param intervals: the passed schedule(vector<pair<int, int> >)
 * @return int: the total amount of jobs that can be completed
 */
int scheduling(std::vector<std::pair<int, int>>& intervals) {
    std::sort(intervals.begin(), intervals.end(),
              [&](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                  if (a.second == b.second) {
                      return a.first < b.first;
                  }
                  return a.second < b.second;
              });

    auto [start_x, start_y] = intervals[0];
    int count_valid = 1;
    for (int i = 1; i < int(intervals.size()); i++) {
        auto [x, y] = intervals[i];
        if (x >= start_y) {
            count_valid++;
            start_x = intervals[i].first;
            start_y = intervals[i].second;
        }
    }

    return count_valid;
}

#endif
