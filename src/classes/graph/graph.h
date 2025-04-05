#ifndef GRAPH_H
#define GRAPH_H

#include <cfloat>
#ifdef ENABLE_GRAPH_VISUALIZATION
#include "../../visualization/graph_visual/graph_visualization.h"
#endif

#ifdef __cplusplus
#include <algorithm>
#include <cassert>
#include <climits>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#endif

/**
 *
 * @brief Class for Unweighted Graph
 *
 */
template <typename T> class graph {
  public:
    /**
     * @brief Constructor for the unweighted graph.
     * @param __type: type of the graph, either "directed" or "undirected"
     * @param __adj: vector<pair<T,vector<T>>, you can pass a vector of pairs to
     * construct the graph without doing multiple add_edge.
     */
    graph(std::string _type, std::vector<std::pair<T, std::vector<T>>> _adj = {}) {
        try {
            if (_type == "directed" || _type == "undirected") {
                this->_type = _type;
            } else {
                throw std::invalid_argument("Can't recognize the type of graph");
            }
            if (!_adj.empty()) {
                for (size_t i = 0; i < _adj.size(); i++) {
                    for (T& neigh : _adj[i].second) {
                        this->add_edge(_adj[i].first, neigh);
                    }
                }
            }
        } catch (std::invalid_argument& e) {
            std::cerr << e.what() << '\n';
            return;
        }
    }

    /**
     * @brief Construct a new graph object
     *
     * @param g the graph we want to copy
     */
    graph(const graph& g) : adj(g.adj), _elements(g._elements), _type(g._type) {}

    /**
     * @brief operator = for the graph class
     *
     * @param g the graph we want to copy
     * @return graph&
     */
    graph& operator=(const graph& g) {
        adj = g.adj;
        _elements = g._elements;
        _type = g._type;
        return *this;
    }

    /**
     * @brief Destroy the graph object
     */
    ~graph() { adj.clear(); }

    /**
     * @brief add_edge function
     * @param u: first node
     * @param v: second node
     */
    void add_edge(T u, T v) {
        if (_type == "undirected") {
            adj[u].push_back(v);
            adj[v].push_back(u);
        } else {
            adj[u].push_back(v);
        }
        _elements.insert(u);
        _elements.insert(v);
    }

    /**
        *@brief has_edge function.
        *@param start: starting node.
        *@param end: ending node.
        *@returns true if a direct edge from start to end exists.
        false if a direct edge from start to end does not exist.
        */
    bool has_edge(T start, T end) {
        if (_elements.find(start) == _elements.end()) {
            return false;
        }
        for (T& x : adj[start]) {
            if (x == end) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief clear function
     * Clearing the entire graph.
     */
    void clear() {
        _elements.clear();
        adj.clear();
    }

    /**
     * @brief empty function
     * Checks if a graph is empty.
     */
    bool empty() { return _elements.empty(); }

    /**
     * @brief size function
     * @returns size_t the size of the graph.
     */
    size_t size();

    /**
     * @brief dfs function
     * @param start: starting node of the dfs.
     * @returns vector<T>, the path of the dfs.
     */
    std::vector<T> dfs(T start);

    /**
     * @brief bfs function
     * @param start: starting node of the bfs.
     * @returns vector<T>, the path of the bfs.
     */
    std::vector<T> bfs(T start);

    /**
     * @brief connected_components function.
     * @returns the connected components(islands) of the graph.
     */
    int64_t connected_components();

    /**
     * @brief cycle function.
     * @returns true if a cycle exist in the graph.
     * */
    bool cycle();

    /**
     * @brief topological_sort function.
     * @returns vector<T> the topological order of the elements of the graph.
     */
    std::vector<T> topological_sort();

    /**
     * @brief bipartite function.
     * @returns true if the graph is bipartite.
     */
    bool bipartite();

    /**
     *@brief bridge function.
     *@param start: starting point of search for the bridges.
     *@returns vector<vector<T>> the bridges of the graph.
     */

    std::vector<std::vector<T>> bridge(T start);

    /**
     *@brief scc(strongly connected components) function.
     *@returns int64_t the number of scc's in the graph using kosaraju's
     *algorithm.
     */
    int64_t scc();

    /**
     *@brief connected function.
     *@returns true if a graph is connected.
     */
    bool connected();

    /**
     *@brief eulerian function.
     *@returns 0 if a graph is not eulerian.
     *         1 if a graph is semi-eulerian.
     *         2 if a graph is eulerian.
     */
    int eulerian();

    /**
     * @brief visualize function.
     * @returns .dot file that can be previewed in vscode with graphviz.
     */
    void visualize();

    /**
     * @brief operator << for the graph class.
     * @returns ostream &out for std::cout.
     */
    friend std::ostream& operator<<(std::ostream& out, graph<T>& g) {
        out << '{';

        std::vector<T> elements = g.topological_sort();
        for (T& x : elements) {
            out << x << ' ';
        }
        out << '}' << '\n';
        return out;
    }

  private:
    /**
     * @param adj: adjacency list for the graph.
     * @param __elements: set of the total elements of the graph.
     * @param __type: the type of the graph, either "directed" or "undirected".
     */
    std::unordered_map<T, std::vector<T>> adj;
    std::unordered_set<T> _elements;
    std::string _type;

    /**
     *@brief helper function for bridge detection algorithm.
     */
    void dfs_bridge(T start, T parent, int64_t& time, std::unordered_map<T, bool>& visited,
                    std::unordered_map<T, int64_t>& in, std::unordered_map<T, int64_t>& out,
                    std::vector<std::vector<T>>& bridges) {
        visited[start] = true;
        in[start] = out[start] = time++;
        for (T& x : adj[start]) {
            if (x != parent) {
                if (visited.find(x) == visited.end()) {
                    dfs_bridge(x, start, time, visited, in, out, bridges);
                    if (out[x] > in[start]) {
                        bridges.push_back({x, start});
                    }
                }
                out[start] = std::min(out[start], out[x]);
            }
        }
    }

    /**
     * @brief helper dfs function for kosaraju's scc
     */
    void dfs_scc(T start, std::unordered_map<T, bool>& visited, std::stack<T>& s) {
        visited[start] = true;
        for (auto& x : adj[start]) {
            if (visited.find(x) == visited.end()) {
                dfs_scc(x, visited, s);
            }
        }

        s.push(start);
    }
};

template <typename T> size_t graph<T>::size() {
    return _elements.size();
}

template <typename T> std::vector<T> graph<T>::dfs(T start) {
    std::vector<T> path;
    if (this->empty() || _elements.find(start) == _elements.end()) {
        return path;
    }
    std::stack<T> s;
    std::unordered_map<T, bool> visited;
    s.push(start);
    visited[start] = true;
    while (!s.empty()) {
        T current = s.top();
        path.push_back(current);
        s.pop();
        for (T& x : adj[current]) {
            if (visited.find(x) == visited.end()) {
                s.push(x);
                visited[x] = true;
            }
        }
    }
    return path;
}

template <typename T> std::vector<T> graph<T>::bfs(T start) {
    std::vector<T> path;
    if (this->empty() || _elements.find(start) == _elements.end()) {
        return path;
    }
    std::queue<T> q;
    std::unordered_map<T, bool> visited;
    q.push(start);
    visited[start] = true;
    while (!q.empty()) {
        int64_t size = q.size();
        for (int64_t i = 0; i < size; i++) {
            T current = q.front();
            path.push_back(current);
            q.pop();
            for (T& x : adj[current]) {
                if (visited.find(x) == visited.end()) {
                    q.push(x);
                    visited[x] = true;
                }
            }
        }
    }
    return path;
}

template <typename T> int64_t graph<T>::connected_components() {
    auto explore = [&](std::unordered_map<T, bool>& visited, T element) -> void {
        std::queue<T> q;
        q.push(element);
        visited[element] = true;
        while (!q.empty()) {
            T current = q.front();
            q.pop();
            for (T& x : adj[current]) {
                if (visited.find(x) == visited.end()) {
                    q.push(x);
                    visited[x] = true;
                }
            }
        }
    };
    size_t n = _elements.size();
    int64_t cc = 0;
    std::unordered_map<T, bool> visited;
    for (T x : _elements) {
        if (visited.find(x) == visited.end()) {
            explore(visited, x);
            cc++;
        }
    }
    return cc;
}

template <typename T> bool graph<T>::cycle() {
    std::unordered_map<T, int> indeg;
    std::queue<T> q;
    size_t visited = 0;

    for (T x : _elements) {
        for (T& y : adj[x]) {
            indeg[y]++;
        }
    }

    for (T x : _elements) {
        if (indeg[x] == 0) {
            q.push(x);
        }
    }

    while (!q.empty()) {
        T current = q.front();
        q.pop();
        visited++;
        for (T& x : adj[current]) {
            if (--indeg[x] == 0) {
                q.push(x);
            }
        }
    }
    return visited == 0;
}

template <typename T> std::vector<T> graph<T>::topological_sort() {
    std::vector<T> top_sort;
    std::unordered_map<T, bool> visited;
    std::unordered_map<T, int64_t> indeg;
    for (T x : _elements) {
        for (T& y : adj[x]) {
            indeg[y]++;
        }
    }

    std::queue<T> q;
    for (T x : _elements) {
        if (indeg[x] == 0) {
            q.push(x);
            visited[x] = true;
        }
    }

    while (!q.empty()) {
        T current = q.front();
        q.pop();
        top_sort.push_back(current);
        for (T& x : adj[current]) {
            if (visited.find(x) == visited.end()) {
                if (--indeg[x] == 0) {
                    q.push(x);
                    visited[x] = true;
                }
            }
        }
    }

    return top_sort;
}

template <typename T> bool graph<T>::bipartite() {
    std::unordered_map<T, int> color;
    std::queue<std::pair<T, int>> q;

    for (T x : _elements) {
        if (color.find(x) == color.end()) {
            q.push({x, 0});
            color[x] = 0;
            while (!q.empty()) {
                std::pair<T, int> current = q.front();
                q.pop();
                T v = current.first;
                int col = current.second;
                for (T& x : adj[v]) {
                    if (color.find(x) != color.end() && color[x] == col) {
                        return false;
                    }
                    if (color.find(x) == color.end()) {
                        color[x] = (col) ? 0 : 1;
                        q.push({x, color[x]});
                    }
                }
            }
        }
    }
    return true;
}

template <typename T> std::vector<std::vector<T>> graph<T>::bridge(T start) {
    int64_t timer = 0;
    std::vector<std::vector<T>> bridges;
    std::unordered_map<T, bool> visited;
    std::unordered_map<T, int64_t> in, out;
    for (T x : _elements) {
        in[x] = 0;
        out[x] = 0;
    }
    dfs_bridge(start, -1, timer, visited, in, out, bridges);
    return bridges;
}

template <typename T> bool graph<T>::connected() {
    std::unordered_map<T, bool> visited;
    bool check = 0;
    T start;
    for (auto& ele : adj) {
        if (adj[ele.first].size() != 0) {
            start = ele.first;
            check = 1;
            break;
        }
    }
    if (!check) {
        return false;
    }
    std::stack<T> s;
    s.push(start);
    visited[start] = true;
    while (!s.empty()) {
        T current = s.top();
        s.pop();
        for (T& x : adj[current]) {
            if (visited.find(x) == visited.end()) {
                visited[x] = true;
                s.push(x);
            }
        }
    }

    for (T x : _elements) {
        if (visited.find(x) == visited.end() && adj[x].size() > 0) {
            return false;
        }
    }
    return true;
}

template <typename T> int64_t graph<T>::scc() {
    if (this->size() == 0) {
        return 0;
    }

    std::unordered_map<T, bool> visited;
    std::stack<T> s;

    for (const T& x : _elements) {
        if (visited.find(x) == visited.end()) {
            dfs_scc(x, visited, s);
        }
    }

    std::unordered_map<T, std::vector<T>> new_adj;
    for (const T& x : _elements) {
        for (const auto& neigh : adj[x]) {
            new_adj[neigh].push_back(x);
        }
    }

    int64_t scc = 0;
    visited.clear();

    auto dfs_new = [&](T start) -> void {
        std::stack<T> _s;
        _s.push(start);
        visited[start] = true;
        while (!_s.empty()) {
            T current = _s.top();
            _s.pop();
            for (auto& x : new_adj[current]) {
                if (visited.find(x) == visited.end()) {
                    _s.push(x);
                    visited[x] = true;
                }
            }
        }
    };

    while (!s.empty()) {
        T current = s.top();
        s.pop();
        if (visited.find(current) == visited.end()) {
            dfs_new(current);
            scc++;
        }
    }

    return scc;
}

template <typename T> int graph<T>::eulerian() {
    if (this->connected() == false) {
        return false;
    }

    int64_t odd = 0;
    for (auto& el : adj) {
        if (adj[el.first].size() & 1) {
            odd++;
        }
    }

    if (odd > 2) {
        return false;
    }
    return (odd) ? 1 : 2;
}

#ifdef ENABLE_GRAPH_VISUALIZATION
template <typename T> void graph<T>::visualize() {
    std::string s;
    if (_type == "directed") {
        if (std::is_same_v<T, char> || std::is_same_v<T, std::string>) {
            for (auto& [element, neighbors] : adj) {
                for (T& x : neighbors) {
                    s += element;
                    s += "->";
                    s += x;
                    s += '\n';
                }
            }
        } else {
            for (auto& [element, neighbors] : adj) {
                for (T& x : neighbors) {
                    s += std::to_string(element);
                    s += "->";
                    s += std::to_string(x);
                    s += '\n';
                }
            }
        }
    } else {
        if (std::is_same_v<T, char> || std::is_same_v<T, std::string>) {
            for (auto& [element, neighbors] : adj) {
                for (T& x : neighbors) {
                    s += element;
                    s += "--";
                    s += x;
                    s += '\n';
                }
            }
        } else {
            for (auto& [element, neighbors] : adj) {
                for (T& x : neighbors) {
                    s += std::to_string(element);
                    s += "--";
                    s += std::to_string(x);
                    s += '\n';
                }
            }
        }
    }
    s += '\n';
    if (_type == "directed") {
        digraph_visualization::visualize(s);
    } else {
        graph_visualization::visualize(s);
    }
}
#endif

/**
 * @brief class for weighted graph
 */
template <typename T> class weighted_graph {
  public:
    /**
     * @brief Constructor for weighted graph.
     * @param __type: type of the graph, either "directed" or "undirected".
     * @param __adj: vector<pair<pair<T,T>, int64_t>>, you can pass a vector of
     * pairs to construct the graph without doing multiple add_edge.
     */
    weighted_graph(std::string _type, std::vector<std::pair<std::pair<T, T>, int64_t>> _adj = {}) {
        try {
            if (_type == "directed" || _type == "undirected") {
                this->_type = _type;
            } else {
                throw std::invalid_argument("Can't recognize the type of graph");
            }
            if (!_adj.empty()) {
                for (size_t i = 0; i < _adj.size(); i++) {
                    this->add_edge(_adj[i].first.first, _adj[i].first.second, _adj[i].second);
                }
            }
        } catch (std::invalid_argument& e) {
            std::cerr << e.what() << '\n';
            return;
        }
    }

    /**
     * @brief Copy constructor for weighted graph class
     * @param g the graph we want to copy
     */
    explicit weighted_graph(const weighted_graph& g)
        : adj(g.adj), _elements(g._elements), _type(g._type) {}

    /**
     * @brief operator = for weighted graph class
     * @param g the graph we want to copy
     * @return weighted_graph&
     */
    weighted_graph& operator=(const weighted_graph& g) {
        adj = g.adj;
        _elements = g._elements;
        _type = g._type;
        return *this;
    }

    /**
     * @brief Destroy the weighted graph object
     *
     */
    ~weighted_graph() { adj.clear(); }

    /**
     * @brief add_edge function.
     * @param u: first node.
     * @param v: second node.
     * @param w: weight between u and v.
     */
    void add_edge(T u, T v, int64_t w) {
        if (_type == "undirected") {
            adj[u].push_back(std::make_pair(v, w));
            adj[v].push_back(std::make_pair(u, w));
        } else if (_type == "directed") {
            adj[u].push_back(std::make_pair(v, w));
        }
        _elements.insert(u);
        _elements.insert(v);
    }

    /**
     *@brief has_edge function.
     *@param start: starting node.
     *@param end: ending node.
     *@returns true if a direct edge from start to end exists.
     */
    bool has_edge(T start, T end) {
        if (_elements.find(start) == _elements.end()) {
            return false;
        }
        for (std::pair<T, double>& x : adj[start]) {
            if (x.first == end) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief clear function.
     * Clearing the entire graph.
     */
    void clear() {
        _elements.clear();
        adj.clear();
    }
    /**
     * @brief empty function.
     * *@returns true if the graph is empty.
     */
    bool empty() { return _elements.empty(); }

    /**
     * @brief size function.
     * @returns the size of the graph.
     */
    size_t size();

    /**
     * @brief dfs function.
     * @param start: starting node of the bfs.
     * @returns vector<T>, the path of the dfs.
     */
    std::vector<T> dfs(T start);

    /**
     * @brief bfs function.
     * @param start: starting node of the bfs.
     * @returns vector<T>, the path of the bfs.
     */
    std::vector<T> bfs(T start);

    /**
     * @brief shortest_path function.
     * @param start: starting node.
     * @param end: ending node.
     * @returns int64_t, the total cost of the path.
     */
    double shortest_path(T start, T end);

    /**
     * @brief connected_components function.
     * @returns the connected componenets(islands) of the graph.
     */
    int64_t connected_components();

    /**
     * @brief cycle function.
     * @returns true if a cycle exists in the graph.
     */
    bool cycle();

    /**
     * @brief topological sort function.
     * @returns vector<T>, the topological order of the elements of the graph.
     */
    std::vector<T> topological_sort();

    /**
     * @brief prim function.
     * @param start: starting node.
     * @returns int64_t, the total cost of the minimum spanning tree of the graph.
     */
    int64_t prim(T start);

    /**
     * @brief bipartite function.
     * @returns true if the graph is bipartite.
     */
    bool bipartite();

    /**
     *@brief bridge function.
     *@param start: starting point of search for the bridges.
     *@returns vector<vector<T>> the bridges of the graph.
     */
    std::vector<std::vector<T>> bridge(T start);

    /**
     *@brief scc(strongly connected components) function.
     *@returns int64_t the total scc's of the graph.
     */
    int64_t scc();

    /**
     *@brief connected function.
     *@returns true if a graph is connected.
     */
    bool connected();

    /**
     *@brief eulerian function.
     *@returns 0 if a graph is not eulerian.
     *         1 if a graph is semi-eulerian.
     *         2 if a graph is eulerian.
     */
    int eulerian();

    /**
     *@brief find SSSP and identify negative cycles.
     *@returns unordered_map of SSSP.
     */
    std::unordered_map<T, double> bellman_ford(T start);

    /**
     *@brief maximum flow function
     *@details Returns the maximum flow from a starting node 's' to an ending
     *node(sink) 't'
     *@returns int: the maximum flow from 's' to 't'
     */
    int max_flow(T start, T end);

    /**
     *@brief visualize function.
     *@returns .dot file that can be previewed in vscode with graphviz.
     */
    void visualize();

    /**
     * @brief << operator for the weighted graph class.
     * @returns ostream &out for std::cout.
     */
    friend std::ostream& operator<<(std::ostream& out, weighted_graph<T>& g) {
        out << '{';
        std::vector<T> elements = g.topological_sort();
        for (T& x : elements) {
            out << x << ' ';
        }
        out << '}' << '\n';
        return out;
    }

  private:
    /**
     * @param adj: adjacency list for the graph.
     * @param __type: type of the graph, either "directed" or "undirected".
     * @param __elements: set of total elements of the graph.
     */
    std::unordered_map<T, std::vector<std::pair<T, double>>> adj;
    std::string _type;
    std::unordered_set<T> _elements;

    /**
     *@brief helper function for bridge detection algorithm.
     */
    void dfs_bridge(T start, T parent, int64_t& time, std::unordered_map<T, bool>& visited,
                    std::unordered_map<T, int64_t>& in, std::unordered_map<T, int64_t>& out,
                    std::vector<std::vector<T>>& bridges) {
        visited[start] = true;
        in[start] = out[start] = time++;
        for (std::pair<T, double>& x : adj[start]) {
            if (x.first != parent) {
                if (visited.find(x.first) == visited.end()) {
                    dfs_bridge(x.first, start, time, visited, in, out, bridges);
                    if (out[x.first] > in[start]) {
                        bridges.push_back({x.first, start});
                    }
                }
                out[start] = std::min(out[start], out[x.first]);
            }
        }
    }

    /**
     * @brief helper dfs function for kosaraju's scc
     */
    void dfs_scc(T start, std::unordered_map<T, bool>& visited, std::stack<T>& s) {
        visited[start] = true;
        for (auto& x : adj[start]) {
            if (visited.find(x.first) == visited.end()) {
                dfs_scc(x.first, visited, s);
            }
        }

        s.push(start);
    }
};

template <typename T> size_t weighted_graph<T>::size() {
    return _elements.size();
}

template <typename T> double weighted_graph<T>::shortest_path(T start, T end) {
    if (_elements.find(start) == _elements.end()) {
        std::cout << "Element: " << start << " is not found in the Graph" << '\n';
        return -1;
    }
    if (_elements.find(end) == _elements.end()) {
        std::cout << "Element: " << end << " is not found in the Graph" << '\n';
        return -1;
    }

    if (!cycle() && _type == "directed") {
        std::vector<T> top_sort = topological_sort();
        std::reverse(top_sort.begin(), top_sort.end());
        std::stack<T> s;
        std::unordered_map<T, double> dist;
        for (auto& x : _elements) {
            dist[x] = std::numeric_limits<int64_t>::max();
        }
        dist[start] = 0;
        while (!top_sort.empty()) {
            auto current = top_sort.back();
            top_sort.pop_back();
            if (dist[current] != std::numeric_limits<int64_t>::max()) {
                for (std::pair<T, double>& x : adj[current]) {
                    if (dist[x.first] > dist[current] + x.second) {
                        dist[x.first] = dist[current] + x.second;
                        top_sort.push_back(x.first);
                    }
                }
            }
        }
        return (dist[end] != std::numeric_limits<int64_t>::max()) ? dist[end] : -1;
    } else {
        std::unordered_map<T, double> dist;
        for (auto& x : _elements) {
            dist[x] = std::numeric_limits<int64_t>::max();
        }
        std::priority_queue<std::pair<double, T>, std::vector<std::pair<double, T>>,
                            std::greater<std::pair<double, T>>>
            pq;
        pq.push(std::make_pair(0, start));
        dist[start] = 0;
        while (!pq.empty()) {
            T currentNode = pq.top().second;
            double currentDist = pq.top().first;
            pq.pop();
            for (std::pair<T, double>& edge : adj[currentNode]) {
                if (currentDist + edge.second < dist[edge.first]) {
                    dist[edge.first] = currentDist + edge.second;
                    pq.push(std::make_pair(dist[edge.first], edge.first));
                }
            }
        }
        return (dist[end] != std::numeric_limits<int64_t>::max()) ? dist[end] : -1;
    }
    return -1;
}

template <typename T> std::vector<T> weighted_graph<T>::dfs(T start) {

    std::vector<T> path;
    if (this->empty() || _elements.find(start) == _elements.end()) {
        return path;
    }
    std::stack<T> s;
    std::unordered_map<T, bool> visited;
    s.push(start);
    visited[start] = true;
    while (!s.empty()) {
        int64_t size = s.size();
        for (int64_t i = 0; i < size; i++) {
            T current = s.top();
            path.push_back(current);
            s.pop();
            for (std::pair<T, double>& x : adj[current]) {
                if (visited.find(x.first) == visited.end()) {
                    s.push(x.first);
                    visited[x.first] = true;
                }
            }
        }
    }
    return path;
}

template <typename T> std::vector<T> weighted_graph<T>::bfs(T start) {
    std::vector<T> path;
    if (this->empty() || _elements.find(start) == _elements.end()) {
        return path;
    }
    std::queue<T> q;
    std::unordered_map<T, bool> visited;
    q.push(start);
    visited[start] = true;
    while (!q.empty()) {
        int64_t size = q.size();
        for (int64_t i = 0; i < size; i++) {
            T current = q.front();
            path.push_back(current);
            q.pop();
            for (std::pair<T, double>& x : adj[current]) {
                if (visited.find(x.first) == visited.end()) {
                    q.push(x.first);
                    visited[x.first] = true;
                }
            }
        }
    }
    return path;
}

template <typename T> int64_t weighted_graph<T>::connected_components() {
    auto explore = [&](std::unordered_map<T, bool>& visited, T element) -> void {
        std::stack<T> s;
        s.push(element);
        visited[element] = true;
        while (!s.empty()) {
            T current = s.top();
            s.pop();
            for (std::pair<T, double>& x : adj[current]) {
                if (visited.find(x.first) == visited.end()) {
                    s.push(x.first);
                    visited[x.first] = true;
                }
            }
        }
    };
    size_t n = _elements.size();
    int64_t cc = 0;
    std::unordered_map<T, bool> visited;
    for (T x : _elements) {
        if (visited.find(x) == visited.end()) {
            explore(visited, x);
            cc++;
        }
    }
    return cc;
}

template <typename T> bool weighted_graph<T>::cycle() {
    std::unordered_map<T, int> indeg;
    std::queue<T> q;
    size_t visited = 0;

    for (T x : _elements) {
        for (std::pair<T, double>& y : adj[x]) {
            indeg[y.first]++;
        }
    }

    for (T x : _elements) {
        if (indeg[x] == 0) {
            q.push(x);
        }
    }

    while (!q.empty()) {
        T current = q.front();
        q.pop();
        visited++;
        for (std::pair<T, double>& x : adj[current]) {
            if (--indeg[x.first] == 0) {
                q.push(x.first);
            }
        }
    }
    return visited == 0;
}

template <typename T> std::vector<T> weighted_graph<T>::topological_sort() {
    std::vector<T> top_sort;
    std::unordered_map<T, bool> visited;
    std::unordered_map<T, int64_t> indeg;
    for (T x : _elements) {
        for (std::pair<T, double>& y : adj[x]) {
            indeg[y.first]++;
        }
    }

    std::queue<T> q;
    for (T x : _elements) {
        if (indeg[x] == 0) {
            q.push(x);
            visited[x] = true;
        }
    }

    while (!q.empty()) {
        T current = q.front();
        q.pop();
        top_sort.push_back(current);
        for (std::pair<T, double>& x : adj[current]) {
            if (visited.find(x.first) == visited.end()) {
                if (--indeg[x.first] == 0) {
                    q.push(x.first);
                    visited[x.first] = true;
                }
            }
        }
    }

    return top_sort;
}

template <typename T> int64_t weighted_graph<T>::prim(T _temp) {
    std::priority_queue<std::pair<T, int64_t>, std::vector<std::pair<T, int64_t>>,
                        std::greater<std::pair<T, int64_t>>>
        q;
    std::unordered_map<T, bool> visited;
    int64_t cost = 0;
    q.push(std::make_pair(0, _temp));
    while (!q.empty()) {
        std::pair<T, int64_t> current = q.top();
        q.pop();
        _temp = current.first;
        if (visited.find(_temp) != visited.end()) {
            continue;
        }
        cost += current.second;
        visited[_temp] = true;
        for (std::pair<T, double>& x : adj[current.first]) {
            if (visited.find(x.first) == visited.end()) {
                q.push(x);
            }
        }
    }
    return cost;
}

template <typename T> bool weighted_graph<T>::bipartite() {
    std::unordered_map<T, int> color;
    std::queue<std::pair<T, int>> q;

    for (T x : _elements) {
        if (color.find(x) == color.end()) {
            q.push({x, 0});
            color[x] = 0;
            while (!q.empty()) {
                std::pair<T, int> current = q.front();
                q.pop();
                T v = current.first;
                int col = current.second;
                for (std::pair<T, double>& x : adj[v]) {
                    if (color.find(x.first) != color.end() && color[x.first] == col) {
                        return false;
                    }
                    if (color.find(x.first) == color.end()) {
                        color[x.first] = (col) ? 0 : 1;
                        q.push({x.first, color[x.first]});
                    }
                }
            }
        }
    }
    return true;
}

template <typename T> std::vector<std::vector<T>> weighted_graph<T>::bridge(T start) {
    int64_t timer = 0;
    std::vector<std::vector<T>> bridges;
    std::unordered_map<T, bool> visited;
    std::unordered_map<T, int64_t> in, out;
    for (T x : _elements) {
        in[x] = 0;
        out[x] = 0;
    }
    dfs_bridge(start, -1, timer, visited, in, out, bridges);
    return bridges;
}

template <typename T> int64_t weighted_graph<T>::scc() {
    if (this->size() == 0) {
        return 0;
    }

    std::unordered_map<T, bool> visited;
    std::stack<T> s;

    for (const T& x : _elements) {
        if (visited.find(x) == visited.end()) {
            dfs_scc(x, visited, s);
        }
    }

    std::unordered_map<T, std::vector<T>> new_adj;
    for (const T& x : _elements) {
        for (const auto& neigh : adj[x]) {
            new_adj[neigh.first].push_back(x);
        }
    }

    int64_t scc = 0;
    visited.clear();

    auto dfs_new = [&](T start) -> void {
        std::stack<T> _s;
        _s.push(start);
        visited[start] = true;
        while (!_s.empty()) {
            T current = _s.top();
            _s.pop();
            for (auto& x : new_adj[current]) {
                if (visited.find(x) == visited.end()) {
                    _s.push(x);
                    visited[x] = true;
                }
            }
        }
    };

    while (!s.empty()) {
        T current = s.top();
        s.pop();
        if (visited.find(current) == visited.end()) {
            dfs_new(current);
            scc++;
        }
    }

    return scc;
}

template <typename T> bool weighted_graph<T>::connected() {
    std::unordered_map<T, bool> visited;
    bool check = 0;
    T start;
    for (auto& ele : adj) {
        if (adj[ele.first].size() != 0) {
            start = ele.first;
            check = 1;
            break;
        }
    }
    if (!check) {
        return false;
    }
    std::stack<T> s;
    s.push(start);
    visited[start] = true;
    while (!s.empty()) {
        T current = s.top();
        s.pop();
        for (std::pair<T, double>& x : adj[current]) {
            if (visited.find(x.first) == visited.end()) {
                visited[x.first] = true;
                s.push(x.first);
            }
        }
    }

    for (T x : _elements) {
        if (visited.find(x) == visited.end() && adj[x].size() > 0) {
            return false;
        }
    }
    return true;
}

template <typename T> int weighted_graph<T>::eulerian() {
    if (this->connected() == false) {
        return false;
    }

    int odd = 0;
    for (auto& el : adj) {
        if (adj[el.first].size() & 1) {
            odd++;
        }
    }

    if (odd > 2) {
        return 0;
    }

    return (odd) ? 1 : 2;
}

template <typename T> std::unordered_map<T, double> weighted_graph<T>::bellman_ford(T start) {
    std::unordered_map<T, double> dist;
    // Initialize the distance to all nodes to be infinity
    // except for the starting node which is zero.
    for (auto it : adj) {
        dist[it.first] = std::numeric_limits<double>::infinity();
    }
    dist[start] = 0;

    // Get number of vertices present in the graph
    int v = adj.size();

    // For each vertex, apply relaxation for all the edges
    for (int i = 0; i < v - 1; i++) {
        for (const auto j : adj) {
            for (const auto edge : adj[j.first]) {
                if (dist[j.first] + edge.second < dist[edge.first]) {
                    dist[edge.first] = dist[j.first] + edge.second;
                }
            }
        }
    }

    // Run algorithm a second time to detect which nodes are part
    // of a negative cycle. A negative cycle has occurred if we
    // can find a better path beyond the optimal solution.
    for (int i = 0; i < v - 1; i++) {
        for (const auto j : adj) {
            for (const auto edge : adj[j.first]) {
                if (dist[j.first] + edge.second < dist[edge.first]) {
                    dist[edge.first] = -std::numeric_limits<double>::infinity();
                }
            }
        }
    }

    // Return the array containing the shortest distance to every node
    return dist;
}

#ifdef ENABLE_GRAPH_VISUALIZATION
template <typename T> void weighted_graph<T>::visualize() {
    std::string s;
    if (_type == "directed") {
        if (std::is_same_v<T, char> || std::is_same_v<T, std::string>) {
            for (auto& [element, neighbors] : adj) {
                for (std::pair<T, double>& x : neighbors) {
                    if (x.first == element) {
                        continue;
                    }
                    s += element;
                    s += "->";
                    s += x.first;
                    s += "[label=";
                    s += std::to_string(x.second);
                    s += "]";
                    s += '\n';
                }
            }
        } else {
            for (auto& [element, neighbors] : adj) {
                for (std::pair<T, double>& x : neighbors) {
                    if (x.first == element) {
                        continue;
                    }
                    s += std::to_string(element);
                    s += "->";
                    s += std::to_string(x.first);
                    s += "[label=";
                    s += std::to_string(x.second);
                    s += "]";
                    s += '\n';
                }
            }
        }
    } else {
        if (std::is_same_v<T, char> || std::is_same_v<T, std::string>) {
            for (auto& [element, neighbors] : adj) {
                for (std::pair<T, double>& x : neighbors) {
                    if (x.first == element) {
                        continue;
                    }
                    s += element;
                    s += "--";
                    s += x.first;
                    s += "[label=";
                    s += std::to_string(x.second);
                    s += "]";
                    s += '\n';
                }
            }
        } else {
            for (auto& [element, neighbors] : adj) {
                for (std::pair<T, double>& x : neighbors) {
                    if (x.first == element) {
                        continue;
                    }
                    s += std::to_string(element);
                    s += "--";
                    s += std::to_string(x.first);
                    s += "[label=";
                    s += std::to_string(x.second);
                    s += "]";
                    s += '\n';
                }
            }
        }
    }
    if (_type == "directed") {
        digraph_visualization::visualize(s);
    } else {
        graph_visualization::visualize(s);
    }
}
#endif

#endif
