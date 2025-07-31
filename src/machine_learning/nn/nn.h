#pragma once

#ifdef __cplusplus
#include <cassert>
#include <iostream>
#include <optional>
#include <random>
#include <vector>
#include "../../algorithms/math/multiply.h"
#endif

namespace nn {

/**
 * @brief Linear module. This implementation mostly follows PyTorch's
 * implementation
 */
class Linear {
  private:
    std::vector<std::vector<double>> weight;
    std::optional<double> bias;
    int in_features_;
    int out_features_;

  public:
    /**
     * @brief Default constructor for nn::Linear class
     * @param in_features(int): The input features
     * @param out_features(int): The output features
     * @param bias(bool): If set to true, then bias will be initialized
     *                    with a uniform distribution on U(-1.0, 1.0)
     */
    explicit Linear(int, int, bool bias = false);

    /**
     * @brief forward function: Forwards an input 1D tensor to the network
     * @param input_tensor: 1D vector, the input tensor
     * @return 1D vector(wT * x + bias)
     */
    std::vector<double> forward(std::vector<double> const&);

    /**
     * @brief updates the weight vector by value
     * @param value: double, the value that will be added to weight vector
     */
    void update_weights(std::vector<double> const&, double, double);
};
} // namespace nn

inline nn::Linear::Linear(int in_features, int out_features, bool bias)
    : in_features_(in_features), out_features_(out_features) {
    assert(in_features != 0);
    assert(out_features != 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    this->weight =
        std::vector<std::vector<double>>(out_features, std::vector<double>(in_features, 0.0));
    for (auto& w_vec : this->weight) {
        for (auto& w : w_vec) {
            w = dist(gen);
        }
    }

    if (bias) {
        this->bias = dist(gen);
    } else {
        this->bias = std::nullopt;
    }
}

inline std::vector<double> nn::Linear::forward(std::vector<double> const& input_tensor) {
    std::vector<double> output(out_features_, 0.0);

    for (int i = 0; i < this->out_features_; i++) {
        for (int j = 0; j < this->in_features_; j++) {
            output[i] += weight[i][j] * input_tensor[j];
        }

        if (bias.has_value()) {
            output[i] += bias.value();
        }
    }

    return output;
}

inline void nn::Linear::update_weights(std::vector<double> const& avg_gradients,
                                       double avg_error,
                                       double learning_rate) {
    for (int i = 0; i < this->out_features_; i++) {
        for (int j = 0; j < this->in_features_; j++) {
            weight[i][j] -= learning_rate * avg_gradients[j];
        }

        if (bias.has_value()) {
            bias = bias.value() - learning_rate * avg_error;
        }
    }
}
