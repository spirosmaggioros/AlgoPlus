#ifndef BFS_H
#define BFS_H

#ifdef __cplusplus
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#endif

/**
 * @brief Breadth First Search (BFS) algorithm
 *
 * @param adj The input graph
 * @param start The starting node
 * @param key The key to search for
 * @return true if the key is found
 * @return false otherwise
 */
template <typename T> bool bfs(std::unordered_map<T, std::vector<T>>& adj, T start, T key) {
    std::unordered_set<T> visited;
    std::queue<T> q;

    visited.insert(start);
    q.push(start);
    if (adj.empty()) {
        return -1;
    } else if (start == key)
        return start;

    while (!q.empty()) {
        int64_t size = q.size();
        for (int64_t i = 0; i < size; i++) {
            auto current = q.front();
            if (current == key) {
                return true;
            }
            q.pop();
            for (T x : adj[current]) {
                if (visited.find(x) == visited.end()) {
                    visited.insert(x);
                    q.push(x);
                }
            }
        }
    }
    return false; // Not found
}

#endif
