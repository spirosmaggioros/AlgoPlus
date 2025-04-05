#ifndef DFS_H
#define DFS_H

#ifdef __cplusplus
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#endif

/**
 * @brief Depth First Search (DFS) algorithm
 *
 * @param adj The input graph
 * @param start The starting node
 * @param key The key to search for
 * @return true if the key is found
 * @return false otherwise
 */
template <typename T> T dfs(std::unordered_map<T, std::vector<T>>& adj, T start, T key) {
    if (adj.empty()) {
        return false;
    }

    std::unordered_set<T> visited;
    std::stack<T> s;
    s.push(start);
    while (!s.empty()) {
        auto current = s.top();
        s.pop();
        if (current == key) {
            return true;
        }

        for (T x : adj[current]) {
            if (visited.find(x) == visited.end()) {
                visited.insert(x);
                s.push(x);
            }
        }
    }
    return false;
}

#endif
