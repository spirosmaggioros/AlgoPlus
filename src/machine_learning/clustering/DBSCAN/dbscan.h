#ifndef DBSCAN_H
#define DBSCAN_H

#ifdef __cplusplus
#include <cstdint>
#include <iostream>
#include <map>
#include <math.h>
#include <stdexcept>
#include <utility>
#include <vector>
#endif

/**
 * @brief DBSCAN clustering algorithm class
 */
class DBSCAN {
  private:
    std::vector<std::pair<double, double>> setOfPoints;
    double Eps;
    int64_t MinPts;
    int64_t cluster_id{0};
    std::map<std::pair<double, double>, int64_t> points;

  public:
    /**
     * @brief constructor for the DBSCAN class
     * @param setOfPoints vector<pair<double,double>>: the input dataset
     * @param Eps: the input diameter
     * @param MinPts: the minimum points that a cluster should have to exist
     */
    explicit DBSCAN(std::vector<std::pair<double, double>> setOfPoints, double Eps,
                    int64_t MinPts) noexcept
        : setOfPoints(setOfPoints), Eps(Eps), MinPts(MinPts) {
        // cluster_id is by default noise

        for (size_t i = 0; i < setOfPoints.size(); ++i) {
            if (points.find(setOfPoints[i]) == points.end()) {
                if (ExpandCluster(setOfPoints, setOfPoints[i], cluster_id, Eps, MinPts)) {
                    cluster_id = nextId(cluster_id);
                }
            }
        }
    }

    /**
     * @ brief nextId function
     * @ return int64_t: the next cluster id
     */
    int64_t nextId(int64_t cluster_id);

    /**
     * @brief ExpandCluster function
     * @param setOfPoints: the input dataset
     * @param point: the input point
     * @param cluster_id: the input cluster_id
     * @param Eps: the input diameter
     * @param MinPts: the minimum points that a cluster should have to exist
     *
     */
    bool ExpandCluster(std::vector<std::pair<double, double>> setOfPoints,
                       std::pair<double, double> point, int64_t cluster_id, double Eps,
                       int64_t MinPts);

    /**
     * @brief get_query function
     * @param setOfPoints: the input dataset
     * @param point: the input point
     * @param Eps: the input diameter
     *
     */
    std::vector<std::pair<double, double>>
    get_query(std::vector<std::pair<double, double>> setOfPoints, std::pair<double, double> point,
              double Eps);

    /**
     * @brief dist function
     * @param a: the first input vector
     * @param b: the second input vector
     * @return double: the euclidean distance of vector a and b
     */
    double dist(std::pair<double, double> a, std::pair<double, double> b);

    /**
     * @brief get_clusters function
     * @return map<pair<double,double>>: the points and their assignments
     *
     */
    std::map<std::pair<double, double>, int64_t> get_clusters();

    /**
     * @brief get_noise function
     * @return vector<pair<double,double>>: the total points that exist as noise
     * in the dataset
     */
    std::vector<std::pair<double, double>> get_noise();
};

inline int64_t DBSCAN::nextId(int64_t cluster_id) {
    try {
        if (cluster_id < -1) {
            throw std::logic_error("cluster_id can't be less than -1");
        }
    } catch (std::logic_error& e) {
        std::cerr << e.what() << '\n';
        return -2;
    }
    cluster_id++;
    return cluster_id;
}

bool DBSCAN::ExpandCluster(std::vector<std::pair<double, double>> setOfPoints,
                           std::pair<double, double> point, int64_t cluster_id, double Eps,
                           int64_t MinPts) {
    std::vector<std::pair<double, double>> seeds = get_query(setOfPoints, point, Eps);
    if (seeds.size() < MinPts) {
        // no core point
        points[point] = -1;
        return false;
    } else {
        // we have a core point
        // all the points in seeds are density-reachable from point
        for (auto& x : seeds) {
            points[x] = cluster_id;
        }
        for (auto it = seeds.begin(); it != seeds.end(); it++) {
            if ((*it).first == point.first && (*it).second == point.second) {
                seeds.erase(it);
                break;
            }
        }

        while (!seeds.empty()) {
            auto current = seeds[0];
            std::vector<std::pair<double, double>> result = get_query(setOfPoints, current, Eps);

            if (result.size() >= MinPts) {
                for (size_t i = 0; i < result.size(); i++) {
                    std::pair<double, double> result_p = result[i];
                    if (points.find(result_p) == points.end() || points[result_p] == -1) {
                        if (points.find(result_p) == points.end()) {
                            seeds.push_back(result_p);
                        }
                        points[result_p] = cluster_id;
                    } // unclassified or noise
                }
            }
            for (auto it = seeds.begin(); it != seeds.end(); it++) {
                if ((*it).first == current.first && (*it).second == current.second) {
                    seeds.erase(it);
                    break;
                }
            }
        }
        return true;
    }
    return false;
}

inline std::vector<std::pair<double, double>>
DBSCAN::get_query(std::vector<std::pair<double, double>> setOfPoints,
                  std::pair<double, double> point, double Eps) {
    std::vector<std::pair<double, double>> ans;
    for (size_t i = 0; i < setOfPoints.size(); i++) {
        std::pair<double, double> curr = setOfPoints[i];
        if (dist(point, curr) <= Eps) {
            ans.push_back(curr);
        }
    }
    return ans;
}

inline double DBSCAN::dist(std::pair<double, double> a, std::pair<double, double> b) {
    return sqrt(pow(b.second - a.second, 2) + pow(b.first - a.first, 2));
}

inline std::map<std::pair<double, double>, int64_t> DBSCAN::get_clusters() {
    std::map<std::pair<double, double>, int64_t> ans;
    for (size_t i = 0; i < setOfPoints.size(); i++) {
        if (points[setOfPoints[i]] != -1) {
            ans[setOfPoints[i]] = points[setOfPoints[i]];
        }
    }
    return ans;
}

inline std::vector<std::pair<double, double>> DBSCAN::get_noise() {
    std::vector<std::pair<double, double>> ans;
    for (size_t i = 0; i < setOfPoints.size(); i++) {
        if (points[setOfPoints[i]] == -1) {
            ans.push_back(setOfPoints[i]);
        }
    }
    return ans;
}

#endif
