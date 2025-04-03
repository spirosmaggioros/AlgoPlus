#pragma once

#ifdef __cplusplus
#include <iostream>
#include <vector>
#include <optional>
#include <cassert>
#include <random>
#include "../../algorithms/math/multiply.h"
#endif


namespace nn {
    class Linear {
    private:
        std::vector<std::vector<double> > weight;
        std::optional<double> bias;
        int in_features_;
        int out_features_;

    public:
        /*
         * @brief Default constructor for nn::Linear class 
         * @param in_features(int): The input features
         * @param out_features(int): The output features
         * @param bias(bool): If set to true, then bias will be initialized
         *                    with a uniform distribution on U(-1.0, 1.0)
         */
        
        explicit Linear(int in_features, int out_features, bool bias=false);
        
        /*
        * @brief forward function: Forwards an input 1D tensor to the network
        * @param input_tensor: 1D vector, the input tensor
        * @return 1D vector(wT * x + bias)
        */
        inline std::vector<double> forward(std::vector<double> const& input_tensor);
    };
}

inline nn::Linear::Linear(int in_features, int out_features, bool bias)
             : in_features_(in_features), out_features_(out_features) {
    assert(in_features != 0);
    assert(out_features != 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    this->weight = std::vector<std::vector<double> >(in_features, std::vector<double>(out_features, 0.0));
    for (auto &w_vec: this->weight) {
        for (auto &w: w_vec) {
            w = dist(gen);
        }
    }

    if (bias) {
        this->bias = dist(gen);
    }
    else {
        this->bias = std::nullopt;
    }
}


inline std::vector<double> nn::Linear::forward(std::vector<double> const& input_tensor) {
    std::vector<std::vector<double> > in_tensor_2d_;
    in_tensor_2d_.push_back(input_tensor);
    std::vector<std::vector<double> > mul_ = multiply(in_tensor_2d_, this->weight);
    for (auto &x: mul_) {
        for (auto &v_: x) {
            v_ += this->bias.value_or(0);
        }
    }

    return mul_[0];
}
