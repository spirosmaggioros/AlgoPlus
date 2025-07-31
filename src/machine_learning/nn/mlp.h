#pragma once

#ifdef __cplusplus
#include <cassert>
#include <iostream>
#include <vector>
#include "../activation/activation_functions.h"
#include "../metrics/metrics.h"
#include "nn.h"
#endif

/**
 * @brief Multilayer Perceptron class. Performs binary and categorical
 * classification Uses nn::Linear as a sequential model, follows PyTorch's
 * implementation.
 * TODO: Addition of Conv1d layers
 */
class MLP {
    std::vector<std::vector<double>> data_;
    std::vector<double> labels_;
    std::vector<nn::Linear> seq_;
    double binary_;
    int epochs_;
    double learning_rate_;

  public:
    /**
     * @brief default constructor for MLP class
     * @param data: 2D vector, The input data. As usual, the last element of each
     * sub-vector represents the label of the row
     * @param arch: 1D vector of pairs. Represents the [in_features, out_features]
     * of each layer in the network
     * @param epochs(int): The number of epochs
     * @param learning_rate(double): The learning rate
     */
    explicit MLP(std::vector<std::vector<double>> const&, std::vector<std::pair<int, int>> const,
                 const int epochs = 100, const double learning_rate = 0.001);

    /**
     * @brief fit an MLP on the input data
     */
    void fit();

    /**
     * @brief performs inference
     * @param input: 1D vector, the passed validation data
     * @return double: The classified label
     */
    double predict(std::vector<double> const&);
};

inline MLP::MLP(std::vector<std::vector<double>> const& data,
                std::vector<std::pair<int, int>> const arch, const int epochs,
                const double learning_rate) {
    assert(data.size() > 0);
    assert(epochs > 0);
    assert(learning_rate > 0);
    assert(arch.size() > 0);
    this->epochs_ = epochs;
    this->data_ = data;
    this->learning_rate_ = learning_rate;
    this->binary_ = (arch.back().second == 1) ? true : false;
    for (std::vector<double>& row : this->data_) {
        this->labels_.push_back(row.back());
        row.pop_back();
    }

    for (auto [in_features_, out_features_] : arch) {
        assert(in_features_ > 0);
        assert(out_features_ > 0);
        this->seq_.push_back(nn::Linear(in_features_, out_features_, true));
    }
}

inline void MLP::fit() {
    for (int epoch = 0; epoch < this->epochs_; epoch++) {
        std::vector<double> y_pred;
        for (size_t i = 0; i < this->data_.size(); i++) {
            std::vector<double> out_ = this->data_[i];
            for (nn::Linear& layer : this->seq_) {
                out_ = layer.forward(out_);
            }

            double y_pred_ = (out_[0] > 0.0) ? 1.0 : -1.0;
            y_pred.push_back(y_pred_);
            // TODO: Perform multiclass classification
            // else {
            //     std::vector<double> logits = activation::softmax(out_);
            //     y_pred = std::max_element(logits.begin(), logits.end()) -
            //     logits.begin(); std::cout << y_pred << '\n';
            // }
            double err = y_pred_ - this->labels_[i];

            if (err != 0) {
                for (nn::Linear& layer : this->seq_) {
                    layer.update_weights(this->data_[i], err, this->learning_rate_);
                }
            }
        }
        std::cout << "Epoch: " << epoch + 1 << ": "
                  << "Accuracy: " << metrics::accuracy_score(this->labels_, y_pred)
                  << " | f1_score: " << metrics::f1_score(this->labels_, y_pred)
                  << " | Recall: " << metrics::recall(this->labels_, y_pred)
                  << " | Precision: " << metrics::precision(this->labels_, y_pred) << '\n';
    }
}

inline double MLP::predict(std::vector<double> const& input) {
    assert(input.size() == this->data_[0].size());
    std::vector<double> out_ = input;
    for (nn::Linear& layer : this->seq_) {
        out_ = layer.forward(out_);
    }

    return (out_[0] > 0.0) ? 1.0 : -1.0;
    // else {
    //     std::vector<double> logits = activation::softmax(out_);
    //     return std::max_element(logits.begin(), logits.end()) - logits.begin();
    // }
}
