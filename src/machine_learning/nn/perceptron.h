#pragma once

#ifdef __cplusplus
#include <cassert>
#include <iostream>
#include <vector>
#include <functional>
#include "../activation/activation_functions.h"
#include "../metrics/metrics.h"
#include "../../algorithms/math/multiply.h"
#include "nn.h"
#endif

/**
 * @brief single layer perceptron implementation using nn::Linear
 */
class perceptron {
  private:
    std::vector<std::vector<double>> data_;
    std::vector<double> labels_;
    int epochs_;
    int num_classes_;
    double learning_rate_;
    nn::Linear weights_;

  public:
    /**
     * @brief default constructor for perceptron class
     * @param data: 2D vector, The input data. As usual, the last element of each
     * sub-vector represents the label of the row
     * @param epochs(int): The number of epochs
     * @param learning_rate(double): The learning rate
     */
    explicit perceptron(std::vector<std::vector<double>> const&,
                        const int num_classes,
                        const int epochs = 100,
                        const double learning_rate = 0.001
    );

    /**
     * @brief fit a single perceptron on the input data
     */
    void fit(const int batch_size);

    /**
     * @brief performs inference, classifying to 1 or -1
     * @param input: 1D vector, the passed validation data
     * @return double: 1.0 or -1.0(binary)
     */
    double predict(std::vector<double> const&);
};

inline perceptron::perceptron(std::vector<std::vector<double>> const& data,
                              const int num_classes,
                              const int epochs,
                              const double learning_rate)
    : weights_(data[0].size() - 1, 1, true) {
    assert(data.size() > 0);
    assert(epochs > 0);
    assert(learning_rate > 0);
    assert(num_classes >= 1);
    this->num_classes_ = num_classes;
    this->epochs_ = epochs;
    this->data_ = data;
    this->learning_rate_ = learning_rate;
    for (std::vector<double>& row : this->data_) {
        this->labels_.push_back(row.back());
        row.pop_back();
    }
}

inline void perceptron::fit(
    const int batch_size
) {
    for (int epoch = 0; epoch < this->epochs_; epoch++) {
        std::vector<double> y_preds;
        int curr_batch = 0;
        int s_batch = 0;
        int e_batch = std::min(batch_size, int(this->data_.size()));
        double avg_error = 0.0;
        std::vector<double> batch_err;
        std::vector<std::vector<double> > batch_inputs;
        while (s_batch < int(this->data_.size())) {
            batch_err.clear();
            batch_inputs.clear();
            avg_error = 0.0;

            for (int i = s_batch; i < e_batch; i++) {
                double y_pred_ = (this->weights_.forward(this->data_[i])[0] > 0);
                double err_ = y_pred_ - this->labels_[i];
                batch_err.push_back(err_);
                avg_error += err_;
                batch_inputs.push_back(this->data_[i]);
                y_preds.push_back(y_pred_);
            }
            avg_error /= (e_batch - s_batch);

            std::vector<double> avg_gradients(this->data_[0].size(), 0.0);
            std::vector<std::vector<double> > grad_w(batch_size, std::vector<double>(this->data_[0].size()));
            for (int sample = 0; sample < (e_batch - s_batch); sample++) {
                for (int features = 0; features < int(batch_inputs[0].size()); features++) {
                    avg_gradients[features] += batch_err[sample] * batch_inputs[sample][features];
                }
            }
            for (int features = 0; features < int(avg_gradients.size()); features++) {
                avg_gradients[features] /= (e_batch - s_batch);
            }
            this->weights_.update_weights(avg_gradients, avg_error, this->learning_rate_);

            curr_batch++;
            s_batch = curr_batch * batch_size;
            e_batch = std::min((curr_batch + 1) * batch_size, int(this->data_.size()));
        }

        std::cout << "Epoch: " << epoch + 1 << ": "
                  << "Accuracy: " << metrics::accuracy_score(this->labels_, y_preds)
                  << " | f1_score: " << metrics::f1_score(this->labels_, y_preds)
                  << " | Recall: " << metrics::recall(this->labels_, y_preds)
                  << " | Precision: " << metrics::precision(this->labels_, y_preds) << '\n';
    }
}

inline double perceptron::predict(std::vector<double> const& input) {
    assert(input.size() == this->data_[0].size());
    return (this->weights_.forward(input)[0] > 0);
}
