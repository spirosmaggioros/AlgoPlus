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

/**
* @brief losses namespace that contains a couple of useful losses in machine learning
*
*/
namespace losses {

    /**
    * @brief sigmoid activation function
    * @param x: double, the input parameter
    * @return double: the sigmoid output
    */
    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

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
            double prob = sigmoid(y_hat[i]);
            double clipped_y_hat = std::clamp(prob, eps, 1 - eps);
            bce += (y[i]*log(clipped_y_hat) + (1-y[i])*log(1 - clipped_y_hat));
        }
        return -bce / double(n);
    }

}


#endif
