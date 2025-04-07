#ifndef DISJOINT_SET_H
#define DISJOINT_SET_H

#ifdef __cplusplus
#include <cstdint>
#include <vector>
#endif

/**
 * @brief disjoint set class
 *
 */
class dsu {
  private:
    std::vector<int64_t> p;
    std::vector<int64_t> depth;
    std::vector<int64_t> ssize;
    std::vector<int64_t> max_el;
    std::vector<int64_t> min_el;

  public:
    /**
     * @brief Construct a new dsu object
     *
     * @param n number of elements
     */
    inline explicit dsu(int64_t n) {
        p.assign(n, 0);
        for (int64_t i = 0; i < n; i++) {
            p[i] = i;
        }
        depth.assign(n, 0);
        max_el.assign(n, 0);
        min_el.assign(n, 0);
        ssize.assign(n, 0);
        for (int64_t i = 0; i < n; i++) {
            depth[i] = 0;
            max_el[i] = i;
            min_el[i] = i;
        }
        for (int64_t i = 0; i < n; i++) {
            ssize[i] = 1;
        }
    }

    /**
     * @brief find function
     *
     * @param i the element we want to search
     * @return int64_t the set it exists in
     */
    inline int64_t find(int64_t i) {
        if (p[i] == i) {
            return i;
        }
        return (p[i] = find(p[i]));
    }

    /**
     * @brief join function
     *
     * @param i first element
     * @param j second element
     * union of i and j
     */
    inline void join(int64_t i, int64_t j) {
        if (same(i, j)) {
            return;
        }

        int64_t x = find(i);
        int64_t y = find(j);

        if (depth[x] > depth[y]) {
            std::swap(x, y);
        }
        p[x] = y;

        if (depth[x] == depth[y]) {
            depth[y]++;
        }
        ssize[y] += ssize[x];
        max_el[y] = std::max(max_el[x], max_el[y]);
        min_el[y] = std::min(min_el[x], min_el[y]);
    }

    /**
     * @brief same function
     *
     * @param i first element
     * @param j second element
     * @return true if i and j exists in the same set
     * @return false if i and j does not exist in the same set
     */
    inline bool same(int64_t i, int64_t j) {
        if (find(i) == find(j)) {
            return true;
        }
        return false;
    }

    inline std::vector<int64_t> get(int64_t i) {
        std::vector<int64_t> ans;
        ans.push_back(get_min(i));
        ans.push_back(get_max(i));
        ans.push_back(size(i));
        return ans;
    }

    /**
     * @brief size function
     *
     * @param i element we are looking for
     * @return int64_t the size of the set that i exists in
     */
    inline int64_t size(int64_t i) { return ssize[find(i)]; }

    /**
     * @brief get the maximum element of the set that i exists in
     *
     * @param i the object that we want to search for
     * @return int64_t the maximum element
     */
    inline int64_t get_max(int64_t i) { return max_el[find(i)]; }

    /**
     * @brief get the minimum element of the set that i exists in
     *
     * @param i the object that we want to search for
     * @return int64_t the minimum element
     */
    inline int64_t get_min(int64_t i) { return min_el[find(i)]; }
};

#endif
