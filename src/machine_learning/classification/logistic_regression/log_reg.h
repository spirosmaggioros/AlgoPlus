#ifndef LOG_REG_H
#define LOG_REG_H

#ifdef __cplusplus
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include "../../metrics/metrics.h"
#include "../../activation/activation_functions.h"
#endif

/**
* @brief logistic regression class
* The implementation follow this: https://www.stat.cmu.edu/~ryantibs/advmethods/notes/logreg.pdf
*/
class logistic_regression {
private:
    double learning_rate_;
    double bias_;
    int epochs_;
    std::vector<double> predictors_;
    std::vector<std::vector<double> > data_;
    std::vector<double> labels_;

    /**
    * @brief returns h_theta(x), where x is the input
    * uses metric's sigmoid to compute that.
    */
    double h_theta(const int index) {
        double z = this->bias_;
        for (int i = 0; i<this->predictors_.size(); i++) {
            z += this->data_[index][i] * this->predictors_[i];
        }

        return activation::sigmoid(z);
    }

    /**
    * @brief performs all the computations to output useful metrics each epoch
    */
    void verbose_acc_step_() {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        std::vector<double> y_pred;
        for (int i = 0; i<this->data_.size(); i++) {
            double h = h_theta(i);

            if (h < 0.5) {
                y_pred.push_back(0.0);
            }
            else {
                y_pred.push_back(1.0);
            }
        }

        std::cout << "Accuracy: " << metrics::accuracy_score(this->labels_, y_pred) 
            << " | f1_score: " << metrics::f1_score(this->labels_, y_pred) 
            << " | Recall: " << metrics::recall(this->labels_, y_pred)
            << " | Precision: " << metrics::precision(this->labels_, y_pred)
            << '\n';
    }

public:
    explicit logistic_regression(const std::vector<std::vector<double> > data,
                                 const double lr=0.001,
                                 const double bias=0.001,
                                 int epochs=10
                                 ) {
        assert(!data.empty());
        for (auto & x: data) {
            assert(x.back() == 0 || x.back() == 1);
        }

        this->learning_rate_ = lr;
        this->epochs_ = epochs;
        this->data_ = data;
        this->bias_ = bias;
        for (auto &x : data) {
            this->labels_.push_back(x.back());
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        this->predictors_ = std::vector<double>(data[0].size() - 1);
        for (int i = 0; i<this->predictors_.size(); i++) {
            this->predictors_[i] = dist(gen);
        }
    }

    /**
     * @brief fits the input to the classifier,
     * performs gradient descent using the predictors and learning rate.
     */
    void fit() {
        for (int i = 0; i<this->epochs_; i++) {
            for (size_t j = 0; j<this->data_.size(); j++) {
                double h = h_theta(j);
                for (size_t k = 0; k<(this->data_[0].size() - 1); k++) {
                    double gradient = (h - this->labels_[j]) * this->data_[j][k];
                    this->predictors_[k] = this->predictors_[k] - (this->learning_rate_ * gradient);
                }
            }

            std::cout << "Epoch: " << (i + 1) << ": ";
            verbose_acc_step_();
        }
    }

};

#endif
