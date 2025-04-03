#pragma once

#ifdef __cplusplus
#include <iostream>
#include <vector>
#include <cassert>
#include "nn.h"
#include "../activation/activation_functions.h"
#endif

/*
* @brief single layer perceptron implementation using nn::Linear
* TODO(follow up): MLP implementation
*/
class perceptron {
private:
    std::vector<std::vector<double> > data_;
    std::vector<double> labels_;
    int epochs_;
    double learning_rate_;
    nn::Linear weights_;

public:
    /*
    * @brief default constructor for perceptron class
    * @param data: the input data, as usual, the last item on each sub-vector is the label
    * @param epochs(int): The number of epochs
    * @param learning_rate(double): The learning rate
    */
    explicit perceptron(std::vector<std::vector<double> > const&, const int epochs=100, const double learning_rate=0.001);
    
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

inline perceptron::perceptron(
    std::vector<std::vector<double> > const& data,
    const int epochs,
    const double learning_rate
) : weights_(data[0].size() - 1, 1, true) {
    assert(data.size() > 0);
    assert(epochs > 0);
    assert(learning_rate > 0);
    this->epochs_ = epochs;
    this->data_ = data;
    this->learning_rate_ = learning_rate;
    for (std::vector<double> &row: this->data_) {
        this->labels_.push_back(row.back());
        row.pop_back();
    }
}

inline void perceptron::fit() {
    for (int epoch=0; epoch<this->epochs_; epoch++) {
        int wrong = 0;
        for (size_t i = 0; i<this->data_.size(); i++) {
            double y_pred = (this->weights_.forward(this->data_[i])[0] > 0) ? 1.0 : -1.0;
            double err = y_pred - this->labels_[i];
            if (err != 0) {
                this->weights_.update_weights(this->data_[i], err, this->learning_rate_);
                wrong += 1;
            }
        }

        std::cout << "Epoch: " << epoch + 1 << " classified: " << wrong << " examples wrong" << '\n';
    }
}

inline double perceptron::predict(std::vector<double> const& input) {
    assert(input.size() == this->data_[0].size());
    return (this->weights_.forward(input)[0] > 0) ? 1.0 : -1.0;
}
