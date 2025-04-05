#ifndef SIMPLE_MULTICLASS
#define SIMPLE_MULTICLASS

#ifdef __cplusplus
#include <cassert>
#include <iostream>
#include <random>
#include <vector>
#include "../../metrics/metrics.h"
#endif

class simple_multi_classification {
  private:
    double learning_rate_;
    double bias_;
    int epochs_;
    std::vector<std::vector<double>> predictors_;
    std::vector<std::vector<double>> data_;
    std::vector<double> labels_;

    void verbose_acc_step_() {
        std::vector<double> y_pred;

        for (int i = 0; i < this->data_.size(); i++) {
            std::vector<double> class_probs(this->predictors_.size());

            double denom = 0.0;
            for (int c = 0; c < this->predictors_.size(); c++) {
                class_probs[c] =
                    exp(std::inner_product(this->predictors_[c].begin(), this->predictors_[c].end(),
                                           this->data_[i].begin(), 0.0) +
                        this->bias_);
                denom += class_probs[c];
            }

            for (int c = 0; c < this->predictors_.size(); c++) {
                class_probs[c] /= denom;
            }

            int pred_class = std::distance(
                class_probs.begin(), std::max_element(class_probs.begin(), class_probs.end()));
            y_pred.push_back(pred_class);
        }

        std::cout << "Accuracy: " << metrics::accuracy_score(this->labels_, y_pred)
                  << " | F1 Score: " << metrics::f1_score(this->labels_, y_pred)
                  << " | Recall: " << metrics::recall(this->labels_, y_pred)
                  << " | Precision: " << metrics::precision(this->labels_, y_pred) << '\n';
    }

  public:
    explicit simple_multi_classification(const std::vector<std::vector<double>> data,
                                         const int num_classes = 2, const double lr = 0.001,
                                         const double bias = 0.001, const int epochs = 10) {
        assert(!data.empty());
        this->learning_rate_ = lr;
        this->epochs_ = epochs;
        this->bias_ = bias;

        this->data_.resize(data.size(), std::vector<double>(data[0].size() - 1));
        this->labels_.resize(data.size());

        for (size_t i = 0; i < data.size(); i++) {
            std::copy(data[i].begin(), data[i].end() - 1, this->data_[i].begin());
            this->labels_[i] = data[i].back();
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        this->predictors_ = std::vector<std::vector<double>>(
            num_classes, std::vector<double>(this->data_[0].size() - 1));
        for (auto& weights : this->predictors_) {
            for (double& weight : weights) {
                weight = dist(gen);
            }
        }
    }

    inline void fit() {
        auto phi__ = [&](int class_, int j) -> double {
            double nom = std::exp(std::inner_product(this->predictors_[class_].begin(),
                                                     this->predictors_[class_].end(),
                                                     this->data_[j].begin(), this->bias_));

            double denom = 0.0;
            for (size_t i = 0; i < this->predictors_.size(); i++) {
                denom += std::exp(std::inner_product(this->predictors_[i].begin(),
                                                     this->predictors_[i].end(),
                                                     this->data_[j].begin(), this->bias_));
            }

            return nom / denom;
        };

        for (int epoch = 0; epoch < this->epochs_; epoch++) {
            std::vector<std::vector<double>> gradients(
                this->predictors_.size(), std::vector<double>(this->data_[0].size(), 0.0));

            for (int j = 0; j < this->data_.size(); j++) {
                int true_class = static_cast<int>(this->labels_[j]);
                for (int i = 0; i < this->predictors_.size(); i++) {
                    double phi_ = phi__(i, j);
                    double diff = phi_ - (true_class == i ? 1.0 : 0.0);

                    for (int w = 0; w < this->data_[j].size(); w++) {
                        gradients[i][w] += diff * this->data_[j][w];
                    }
                }
            }

            // perform stochastic gradient descent
            for (int i = 0; i < this->predictors_.size(); i++) {
                for (int w = 0; w < this->data_[0].size(); w++) {
                    this->predictors_[i][w] -=
                        this->learning_rate_ * gradients[i][w] / this->data_.size();
                }
            }

            std::cout << "Epoch: " << (epoch + 1) << ": ";
            verbose_acc_step_();
        }
    }
};

#endif
