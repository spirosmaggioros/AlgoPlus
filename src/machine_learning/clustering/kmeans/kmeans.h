#ifndef KMEANS_H
#define KMEANS_H

#ifdef __cplusplus
#include "../../../../third_party/json.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <vector>
#endif

using json = nlohmann::json;

/**
 * @ brief Class for the kmeans clustering algorithm
 */
class kmeans {
  private:
    /**
     * @ brief distance function
     * @ param a the first input vector
     * @ param b the second input vector
     * @ return int64_t the euclidean distance between vector a and b
     *
     */
    double distance(std::vector<double>& a, std::vector<double>& b) {
        return sqrt(pow((a[0] - b[0]), 2) + pow((a[1] - b[1]), 2));
    }
    std::vector<std::vector<double>> data;
    int K;
    std::vector<std::vector<double>> cluster_centers;
    std::map<std::vector<double>, int64_t> assignments;

  public:
    /**
     * @brief Constructor for the kmeans class
     * @param data: the input data(2D vector)
     * @param K: the number of clusters
     * @param MAX_ITER: default 500, maximum iterations till it converges
     */
    kmeans(std::vector<std::vector<double>> data, int K, int64_t MAX_ITER = 1500)
        : data(data), K(K) {

        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_real_distribution<double> distrib(std::numeric_limits<double>::min(),
                                                       std::numeric_limits<double>::max());
        for (int i = 0; i < K; i++) {
            double rand_num = distrib(gen);
            this->cluster_centers.push_back(data[rand_num]);
        }

        for (int ww = 0; ww < MAX_ITER; ww++) {
            for (int64_t i = 0; i < data.size(); i++) {
                assign_to_closest(data[i]);
            }

            std::vector<std::vector<std::vector<double>>> _clusters(K);
            for (auto& x : assignments) {
                _clusters[x.second].push_back(x.first);
            }
            std::vector<std::vector<double>> new_centroids;
            for (int i = 0; i < K; i++) {
                new_centroids.push_back(get_centroid(_clusters[i]));
            }

            if (new_centroids == this->cluster_centers) {
                break;
            } else {
                this->cluster_centers = new_centroids;
            }
        }
    }

    /**
     * @brief Destroy the kmeans object
     */
    ~kmeans() {
        cluster_centers.clear();
        assignments.clear();
    }

    /**
     * @ brief assign_to_closest function
     * assigns the closest centroid to point x
     * @ param x: the input vector
     */
    inline void assign_to_closest(std::vector<double>& x) {
        std::vector<double> id = this->cluster_centers[0];
        int index = 0;
        int min_dist = distance(x, id);
        for (int j = 0; j < this->cluster_centers.size(); j++) {
            int current_dist = distance(this->cluster_centers[j], x);
            if (current_dist < min_dist) {
                min_dist = current_dist;
                index = j;
                id = this->cluster_centers[index];
            }
        }
        assignments[x] = index;
    }

    /**
     * @ brief get_centroid function
     * @ param cluster: the input cluster
     * @ return vector<int>: the centroid of the cluster
     */
    inline std::vector<double> get_centroid(std::vector<std::vector<double>> cluster) {
        double sum_x = 0, sum_y = 0, n = cluster.size();
        for (auto& x : cluster) {
            sum_x += x[0];
            sum_y += x[1];
        }
        sum_x /= n;
        sum_y /= n;
        return {sum_x, sum_y};
    }

    /**
     * @ brief fit function
     * @ return pair<vector<int64_t>, map<int64_t, int> >: the centroids and the
     * assignments of the clustering algorithm
     *
     */
    inline std::pair<std::vector<std::vector<double>>, std::map<std::vector<double>, int64_t>>
    fit() {
        return std::make_pair(cluster_centers, assignments);
    }
};

#endif
