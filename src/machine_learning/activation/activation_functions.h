#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#ifdef __cplusplus
#include <cmath>
#include <vector>
#endif

namespace activation {
/**
 * @brief sigmoid activation function
 * @param x: double, the input parameter
 * @return double: the sigmoid output
 */
inline double sigmoid(const double x) {
    return 1.0 / (1.0 + exp(-x));
}

/**
 * @brief ReLU activation function
 * @param x(double): the input value
 */
inline double ReLU(const double x) {
    return std::fmax(0, x);
}

/**
 * @brief LeakyReLU activation function
 * @param x(double): the input value
 * @param a(double): the multiplication coefficient for values < 0
 */
inline double LeakyReLU(const double x, const double a = 0.01) {
    if (x < 0) {
        return a * x;
    }
    return x;
}

/**
 * @brief softmax activation function for multiclass classification
 * @param logits(vector<double>): A vector that holds the logits
 */
inline std::vector<double> softmax(const std::vector<double>& logits) {
    double sum_logits = 0;
    for (const double& x : logits) {
        sum_logits += exp(x);
    }

    std::vector<double> probs;
    for (const double& x : logits) {
        probs.push_back(exp(x) / sum_logits);
    }

    return probs;
}
} // namespace activation

#endif
