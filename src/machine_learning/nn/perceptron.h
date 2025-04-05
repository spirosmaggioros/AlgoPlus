#pragma once

#ifdef __cplusplus
#include <cassert>
#include <iostream>
#include <vector>
#include "../activation/activation_functions.h"
#include "../metrics/metrics.h"
#include "nn.h"
#endif

/*
 * @brief single layer perceptron implementation using nn::Linear
 * TODO(follow up): MLP implementation
 */
class perceptron {
  private:
    std::vector<std::vector<double>> data_;
    std::vector<double> labels_;
    int epochs_;
    double learning_rate_;
    nn::Linear weights_;

  public:
    /*
     * @brief default constructor for perceptron class
     * @param data: 2D vector, The input data. As usual, the last element of each
     * sub-vector represents the label of the row
     * @param epochs(int): The number of epochs
     * @param learning_rate(double): The learning rate
     */
    explicit perceptron(std::vector<std::vector<double>> const&, const int epochs = 100,
                        const double learning_rate = 0.001);

    /*
     * @brief fit a single perceptron on the input data
     */
    void fit();

    /*
     * @brief performs inference, classifying to 1 or -1
     * @param input: 1D vector, the passed validation data
     * @return double: 1.0 or -1.0(binary)
     */
    double predict(std::vector<double> const&);
};

inline perceptron::perceptron(std::vector<std::vector<double>> const& data, const int epochs,
                              const double learning_rate)
    : weights_(data[0].size() - 1, 1, true) {
    assert(data.size() > 0);
    assert(epochs > 0);
    assert(learning_rate > 0);
    this->epochs_ = epochs;
    this->data_ = data;
    this->learning_rate_ = learning_rate;
    for (std::vector<double>& row : this->data_) {
        this->labels_.push_back(row.back());
        row.pop_back();
    }
}

inline void perceptron::fit() {
    for (int epoch = 0; epoch < this->epochs_; epoch++) {
        std::vector<double> y_pred;
        for (size_t i = 0; i < this->data_.size(); i++) {
            double y_pred_ = (this->weights_.forward(this->data_[i])[0] > 0) ? 1.0 : -1.0;
            y_pred.push_back(y_pred_);
            double err = y_pred_ - this->labels_[i];
            if (err != 0) {
                this->weights_.update_weights(this->data_[i], err, this->learning_rate_);
            }
        }

        std::cout << "Epoch: " << epoch + 1 << ": "
                  << "Accuracy: " << metrics::accuracy_score(this->labels_, y_pred)
                  << " | f1_score: " << metrics::f1_score(this->labels_, y_pred)
                  << " | Recall: " << metrics::recall(this->labels_, y_pred)
                  << " | Precision: " << metrics::precision(this->labels_, y_pred) << '\n';
    }
}

inline double perceptron::predict(std::vector<double> const& input) {
    assert(input.size() == this->data_[0].size());
    return (this->weights_.forward(input)[0] > 0) ? 1.0 : -1.0;
}
