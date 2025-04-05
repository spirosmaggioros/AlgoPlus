#ifndef SHORTEST_PATH_H
#define SHORTEST_PATH_H

#ifdef __cplusplus
#include <iostream>
#include <limits>
#include <queue>
#include <unordered_map>
#include <vector>
#endif

/**
 * @brief shortest path algorithm. A more CP friendly version to be used in
 * competitions
 * @param adj: the input graph
 * @param n: the total nodes in the graph(this is used for continuous nodes, i.e
 * 1,...,n)
 * @param start: starting node
 * @param end: ending node
 * @return vector<int>: the shortest distances for each node n_i to end
 */
int shortest_path(std::unordered_map<int, std::vector<std::pair<int, int>>>& adj, int n, int start,
                  int end) {
    std::vector<int> dist(n, std::numeric_limits<int>::max());

    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>,
                        std::greater<std::pair<int, int>>>
        q;
    q.push(std::make_pair(0, start));
    dist[start] = 0;

    while (!q.empty()) {
        int currentNode = q.top().second;
        int currentDist = q.top().first;
        q.pop();

        if (currentDist > dist[currentNode]) {
            continue;
        }
        for (std::pair<int, int>& edge : adj[currentNode]) {
            if (currentDist + edge.second < dist[edge.first]) {
                dist[edge.first] = currentDist + edge.second;
                q.push(std::make_pair(dist[edge.first], edge.first));
            }
        }
    }

    return (dist[end] != std::numeric_limits<int>::max()) ? dist[end] : -1;
}

#endif
