#ifndef MEAN_SQUARED_ERROR_H
#define MEAN_SQUARED_ERROR_H

#ifdef __cplusplus
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cassert>
#include <numbers>
#include <numeric>
#endif

namespace _metrics_utils {
    double sigmoid(const double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    std::vector<double> softmax(const std::vector<double> &logits) {
        double sum_logits = 0;
        for (const double &x: logits) {
            sum_logits += exp(x);
        }

        std::vector<double> probs;
        for (const double &x: logits) {
            probs.push_back(exp(x) / sum_logits);
        }

        return probs;
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
        * @return tuple<int, int, int, int>
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
    * @return double
    */
    double recall(const std::vector<double> &y, const std::vector<double> &y_pred) {
        auto [tp, tn, fp, fn] = multi_metrics_::all_metrics_(y, y_pred);
        return 1.0 * (tp) / (tp + fn);
    }

    /**
     * @brief accuracy score function[(tp + tn) / (tp + tn + fp + fn)]
     * @return double
     */
    double accuracy_score(const std::vector<double> &y, const std::vector<double> &y_pred) {
        auto [tp, tn, fp, fn] = multi_metrics_::all_metrics_(y, y_pred);
        return 1.0 * (tp + tn) / (tp + tn + fp + fn);
    }

    /**
     * @brief precision function[tp / tp + fp]
     * @return double
     */
    double precision(const std::vector<double> &y, const std::vector<double> &y_pred) {
        auto [tp, tn, fp, fn] = multi_metrics_::all_metrics_(y, y_pred);
        return 1.0 * tp / (tp + fp);
    }

    /**
     * @brief f1 score function: [2 * precision * recall / precision + recall]
     * @return double
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
     * @return double
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
     * @return double
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
     * @return double
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
        * @return double
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
        * @return double
        */
        double root_mean_squared_error(std::vector<double> const& y, std::vector<double> const& y_hat) {
            return std::sqrt(mean_squared_error(y, y_hat));
        }


        /**
        * @brief mean absolute error function
        * @param y: vector, the original labels
        * @param y_hat: vector, the predicted labels
        * @return double
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

        // /**
        // * @brief log loss for multiclass classification
        // * @param y(vector<double>): the original labels
        // * @param y_hat(vector<double>): the predicted labels
        // * @return double
        // */
        // double xlogy(std::vector<double> const& y, std::vector<double> const& y_hat) {
        //     assert(y.size() == y_hat.size());
        //     std::vector<double> softmax_ = _metrics_utils::softmax(y);
        //     assert(softmax_.size() == y.size());

        //     size_t n = y.size();
        //     double ce = 0.0;
        //     for (const double &x: softmax_) {
        //         ce += log(x);
        //     }
        //     return -ce;
        // }

        /**
        * @brief binary crossentropy loss for binary classification
        * @param y: vector, the original labels
        * @param y_hat: vector, the predicted labels
        * @return double
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
