#ifndef MEAN_SQUARED_ERROR_H
#define MEAN_SQUARED_ERROR_H

#ifdef __cplusplus
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cassert>
#include <numbers>
#endif

namespace _metrics_utils {
    double sigmoid(const double x) {
        return 1.0 / (1.0 + exp(-x));
    }
}

/**
* @brief losses namespace that contains a couple of useful losses in machine learning
*
*/
namespace metrics {
    
    namespace multi_metrics_ {
        /**
        * @brief multi metrics function. Returns tp, tn, fp, fn
        * @param y: the ground truth(vector<double>)
        * @param y_pred: the predictions(vector<double>)
        */
        std::tuple<int, int, int, int> all_metrics_(const std::vector<double> &y, const std::vector<double> &y_pred) {
            assert(y.size() == y_pred.size());
            int tp = 0, tn = 0, fp = 0, fn = 0;
            for (size_t i = 0; i<y.size(); i++) {
                if (y_pred[i] == y[i] && y[i] == 1) {
                    tp += 1;
                }
                else if (y_pred[i] == y[i] && y[i] == 0) {
                    tn += 1;
                }
                else if (y_pred[i] != y[i] && y[i] == 1) {
                    fn += 1;
                }
                else if (y_pred[i] != y[i] && y[i] == 0) {
                    fp += 1;
                }
            }

            return { tp, tn, fp, fn };
        } 
    }

    /**
    * @brief recall function[tp / tp + fn]
    */
    double recall(const std::vector<double> &y, const std::vector<double> &y_pred) {
        auto [tp, tn, fp, fn] = multi_metrics_::all_metrics_(y, y_pred);
        return 1.0 * (tp) / (tp + fn);
    }
    
    /**
     * @brief accuracy score function[(tp + tn) / (tp + tn + fp + fn)]
     */
    double accuracy_score(const std::vector<double> &y, const std::vector<double> &y_pred) {
        auto [tp, tn, fp, fn] = multi_metrics_::all_metrics_(y, y_pred);
        return 1.0 * (tp + tn) / (tp + tn + fp + fn);
    }
    
    /**
     * @brief precision function[tp / tp + fp]
     */
    double precision(const std::vector<double> &y, const std::vector<double> &y_pred) {
        auto [tp, tn, fp, fn] = multi_metrics_::all_metrics_(y, y_pred);
        return 1.0 * tp / (tp + fp);
    }
    
    /**
     * @brief f1 score function: [2 * precision * recall / precision + recall]
     */
    double f1_score(const std::vector<double> &y, const std::vector<double> &y_pred) {
        auto [tp, tn, fp, fn] = multi_metrics_::all_metrics_(y, y_pred);
        double prec = precision(y, y_pred), rec = recall(y, y_pred);
        return 2.0 * (prec * rec) / (prec + rec);
    }

    /**
     * @brief euclidean distance function
     * @param x(vector<double>): the first passed vector
     * @param y(vector<double>): the second passed vector
     */
    double euclidean_distance(const std::vector<double> &x, const std::vector<double> &y) {
        assert(x.size() == y.size());

        double _dist = 0.0;
        for (size_t i = 0; i<x.size(); i++) {
            _dist += std::powf(y[i] - x[i], 2);
        }

        return std::sqrt(_dist);
    }

    /**
     * @brief manhattan distance function
     * @param x(vector<double>): the first passed vector
     * @param y(vector<double>): the secoond passed vector
     */
    double manhattan_distance(const std::vector<double> &x, const std::vector<double> &y) {
        assert(x.size() == y.size());

        double _dist = 0.0;
        for (size_t i = 0; i<x.size(); i++) {
            _dist += std::abs(y[i] - x[i]);
        }

        return _dist;
    }

    
    /**
     * @brief minkowski distance
     * @param x(vector<double>): the first passed vector
     * @param y(vector<double>): the second passed vector
     * @param p(double): The order of the norm of the difference
     */
    double minkowski_distance(const std::vector<double> &x, const std::vector<double> &y, const double p) {
        assert(x.size() == y.size());

        double _dist = 0.0;
        for (size_t i = 0; i<x.size(); i++) {
            _dist += std::abs(y[i] - x[i]);
        }

        return std::powf(_dist, 1.0 / p);
    }


    namespace losses {
        /**
        * @brief mean squared error function
        * @param y: vector, the original labels
        * @param y_hat: vector, the predicted labels
        * @return double: the mean squared error
        */
        double mean_squared_error(std::vector<double> const& y, std::vector<double> const& y_hat) {
            assert(y.size() == y_hat.size());
            size_t n = y.size();
            double mse = 0.0;
            for(size_t i = 0; i<n; i++) {
                mse += powf(y[i] - y_hat[i], 2);
            }
            return mse / double(n);
        }

        /**
        * @brief root mean squared error function
        * @param y: vector, the original labels
        * @param y_hat: vector, the predicted labels
        * @return double: the root mean squared error
        */
        double root_mean_squared_error(std::vector<double> const& y, std::vector<double> const& y_hat) {
            return std::sqrt(mean_squared_error(y, y_hat));
        }


        /**
        * @brief mean absolute error function
        * @param y: vector, the original labels
        * @param y_hat: vector, the predicted labels
        * @return double: the mean absolute error
        */
        double mean_absolute_error(std::vector<double> const& y, std::vector<double> const& y_hat) {
            assert(y.size() == y_hat.size());
            size_t n = y.size();
            double mae = 0.0;
            for(size_t i = 0; i<n; i++) {
                mae += std::abs(y[i] - y_hat[i]);
            }
            return mae / double(n);
        }

        /**
        * @brief binary crossentropy loss function
        * @param y: vector, the original labels
        * @param y_hat: vector, the predicted labels
        * @return double: the binary crossentropy loss
        */
        double binary_crossentropy_loss(std::vector<double> const& y, std::vector<double> const& y_hat) {
            assert(y.size() == y_hat.size());
            for(auto & x : y) {
                assert(x == 0.0 || x == 1.0);
            }

            size_t n = y.size();
            double bce = 0.0, eps = 1e-15;
            for(size_t i = 0; i<n; i++) {
                double prob = _metrics_utils::sigmoid(y_hat[i]);
                double clipped_y_hat = std::clamp(prob, eps, 1 - eps);
                bce += (y[i]*log(clipped_y_hat) + (1-y[i])*log(1 - clipped_y_hat));
            }
            return -bce / double(n);
        }

    }
}


#endif
