#include <queue>
#ifndef ASTAR_H
#define ASTAR_H

#ifdef __cplusplus
#include <algorithm>
#include <climits>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>
#endif

/**
 * @ brief A* Class
 */
template <typename T> class AStar {
  private:
    std::unordered_map<T, std::vector<std::pair<T, double>>> adj;
    std::unordered_map<T, double> nodes;

  public:
    /**
     * @brief A* constructor
     * @param v: unordered_map<T, vector<pair<T, double> > > initializer adjacency
     * list. Default = {}
     * @param nodes: unordered_map<T, double> contains the heuristic value of each
     * node. Default = {}
     */
    explicit AStar(std::unordered_map<T, std::vector<std::pair<T, double>>> v = {},
                   std::unordered_map<T, double> nodes = {}) {
        try {
            if ((!v.empty() && nodes.empty()) || (!nodes.empty() && v.empty())) {
                throw std::logic_error("you have to provide two non empty maps");
            }
            if (!v.empty() && !nodes.empty()) {
                this->adj = v;
                this->nodes = nodes;
            }
        } catch (std::logic_error& e) {
            std::cerr << e.what() << '\n';
        }
    }

    /**
     * @brief insert_node function
     * @param u: the node ID
     * @val: the heuristic value of node u
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
     * @param dist: the distance between u and v
     */
    inline void add_edge(T u, T v, double dist) {
        try {
            if (nodes.find(u) != nodes.end() && nodes.find(v) != nodes.end()) {
                adj[u].push_back(std::make_pair(v, dist));
            } else {
                throw std::logic_error("One of the two nodes that passed to the "
                                       "function do not exist in the graph");
            }
        } catch (std::logic_error& e) {
            std::cerr << e.what() << '\n';
        }
    }

    /**
     * @brief shortest_path function
     * @param start: starting node
     * @param end: end node
     * @return vector<T>: the shortest path from start to end
     */
    inline std::vector<T> shortest_path(T start, T end) {
        auto reconstruct_path = [&](std::unordered_map<T, T> cameFrom, T curr) {
            std::vector<T> path = {curr};
            while (cameFrom.find(curr) != cameFrom.end()) {
                curr = cameFrom[curr];
                path.push_back(curr);
            }
            std::reverse(path.begin(), path.end());
            return path;
        };
        std::priority_queue<std::pair<double, T>, std::vector<std::pair<double, T>>,
                            std::greater<std::pair<double, T>>>
            pq;
        std::unordered_map<T, bool> visited;
        std::unordered_map<T, T> cameFrom;
        std::unordered_map<T, double> gScore;
        std::unordered_map<T, double> fScore;
        for (auto& x : nodes) {
            gScore[x.first] = INT_MAX;
            fScore[x.first] = INT_MAX;
        }
        gScore[start] = 0;
        fScore[start] = nodes[start];
        pq.push(std::make_pair(0, start));
        visited[start] = true;
        while (!pq.empty()) {
            auto current = pq.top();
            if (current.second == end) {
                return reconstruct_path(cameFrom, current.second);
            }
            pq.pop();
            for (auto& x : adj[current.second]) {
                double tentative_gscore = gScore[current.second] + x.second;
                if (tentative_gscore < gScore[x.first]) {
                    cameFrom[x.first] = current.second;
                    gScore[x.first] = tentative_gscore;
                    fScore[x.first] = tentative_gscore + nodes[x.first];
                    if (visited.find(x.first) == visited.end()) {
                        pq.push(std::make_pair(tentative_gscore + nodes[x.first], x.first));
                    }
                }
            }
        }
        return {};
    }
};

#endif
