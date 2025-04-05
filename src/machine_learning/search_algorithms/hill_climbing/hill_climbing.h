#ifndef HILL_CLIMBING_H
#define HILL_CLIMBING_H

#ifdef __cplusplus
#include <climits>
#include <iostream>
#include <limits>
#include <queue>
#include <unordered_map>
#endif

/**
 * @brief hill climbing class
 */
template <typename T> class hill_climbing {
  private:
    std::unordered_map<T, std::vector<std::pair<T, double>>> adj;
    std::unordered_map<T, double> nodes;

  public:
    /**
     * @brief hill_climbing constructor
     * @param v: unordered_map<T, pair<T, double> > initializer vector. Default =
     * {}
     * @param nodes: unordered_map<T, double> initializer values for the nodes.
     * Default = {}
     */
    explicit hill_climbing(std::unordered_map<T, std::vector<std::pair<T, double>>> v = {},
                           std::unordered_map<T, double> nodes = {}) {
        if (!v.empty()) {
            this->adj = v;
        }
        if (!nodes.empty()) {
            this->nodes = nodes;
        }
    }

    /**
     * @brief insert_node function
     * @param u: the node ID
     * @param val: the heuristic value of node u
     */
    inline void insert_node(T u, double val) { nodes[u] = val; }

    /**
     * @brief has_edge function
     * @param u: the first node
     * @param v: the second node
     * @return true if there exist an edge between u and v
     * @return false otherwise
     */
    inline bool has_edge(T u, T v) {
        if (adj.find(u) != adj.end()) {
            for (std::pair<T, double>& x : adj[u]) {
                if (x.first == v) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * @brief add_edge function
     * @param u: the first node
     * @param v: the second node
     */
    inline void add_edge(T u, T v) {
        try {
            if (nodes.find(u) != nodes.end() && nodes.find(v) != nodes.end()) {
                adj[u].push_back(std::make_pair(v, nodes[v]));
            } else {
                throw std::logic_error("One of the two nodes that passed to the "
                                       "function do not exist in the graph");
            }
        } catch (std::logic_error& e) {
            std::cerr << e.what() << '\n';
        }
    }

    /**
     * @brief search function
     * @param start: starting node
     * @param end: end node
     * @return true if search found the end node.
     * @return false otherwise
     */
    inline bool search(T start, T end) {
        if (adj.empty()) {
            return false;
        }
        std::unordered_map<T, bool> visited;
        std::queue<T> q;
        q.push(start);
        visited[start] = true;
        while (!q.empty()) {
            int64_t size = q.size();
            for (int64_t i = 0; i < size; i++) {
                T current = q.front();
                if (current == end) {
                    return true;
                }
                q.pop();
                double minim = std::numeric_limits<double>::max();
                T neigh = current;
                for (std::pair<T, double>& x : adj[current]) {
                    if (visited.find(x.first) == visited.end()) {
                        visited[x.first] = true;
                        minim = std::min(minim, x.second);
                        neigh = x.first;
                    }
                }
                if (minim != INT_MAX && neigh != current) {
                    if (current == start || minim <= nodes[current]) {
                        q.push(neigh);
                    }
                }
            }
        }
        return false;
    }
};

#endif
