#ifndef MINIMUM_SWAPS_H
#define MINIMUM_SWAPS_H

#ifdef __cplusplus
#include <iostream>
#include <vector>
#include <cassert>
#include <map>
#include <algorithm>
#endif

/**
* @brief minimum swaps function.
* @details computes the minimum swaps to make array two arrays equal
* @param a: vector<int>
* @param b: vector<int>
* @return int: the minimum number of swaps to make array a equal to array b
*/
int minimum_swaps(std::vector<int> &a, std::vector<int> &b) {
    assert(a.size() == b.size());
    int n = int(a.size());
    auto solve = [&](std::vector<int> &arr) -> int {
        std::vector<std::pair<int, int> > positions;
        for(int i = 0; i<n; i++) {
            int pos = i, value = arr[i];
            positions.push_back({value, pos});
        }

        std::sort(positions.begin(), positions.end());

        std::vector<bool> visited(n, false);

        int min_swaps = 0;
        for(int i = 0; i<n; i++) {
            if(visited[i] || positions[i].second == i) {
                continue;
            }

            int cycle_size = 0, j = i;
            while(!visited[j]) {
                visited[j] = 1;
                j = positions[j].second;
                cycle_size++;
            }

            min_swaps += (cycle_size - 1);
        }

        return min_swaps;
    };

    std::map<int, int>  m;
    for(int i = 0; i<n; i++) {
        m[b[i]] = i;
    }

    for(int i = 0; i<n; i++) {
        b[i] = m[a[i]];
    }

    return solve(b);
}



#endif
