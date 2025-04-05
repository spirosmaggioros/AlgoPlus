#ifndef LIN_REG_H
#define LIN_REG_H

#ifdef __cplusplus
#include <cmath>
#include <iostream>
#include <vector>
#endif

/**
 * @brief Class for linear regression algorithm
 */
class linear_regression {
  private:
    std::vector<std::vector<double>> data;
    // for y = a + b*x
    double a, b;
    double Sxy, Sxx;
    double x_mean, y_mean;

  public:
    /**
     * @brief Constructor for linear regression class
     * @param points: 2D points that user pass to the regressor
     */
    explicit linear_regression(std::vector<std::vector<double>> points) {
        try {
            if (!points.empty()) {
                this->data = points;
                Sxy = 0.0;
                Sxx = 0.0;
                a = 0;
                b = 0;
                x_mean = 0;
                y_mean = 0;
            } else {
                throw std::logic_error("Data array is empty!");
            }
        } catch (std::logic_error& e) {
            std::cerr << e.what() << '\n';
        }
    }

    /**
     * @brief get_results function
     * @return pair<double,double> the values of a and b
     *
     */
    inline std::pair<double, double> get_results() {
        int64_t n = this->data.size();
        for (auto& x : data) {
            x_mean += x[0];
            y_mean += x[1];
        }
        x_mean /= n;
        y_mean /= n;
        for (auto& x : data) {
            Sxy += (x[0] - this->x_mean) * (x[1] - this->y_mean);
            Sxx += std::pow((x[0] - this->x_mean), 2);
        }
        b = Sxy / Sxx;
        a = y_mean - b * x_mean;
        return std::make_pair(a, b);
    }

    /**
     * @brief predict function
     * @param x: the value of x which we want to predict y
     * @return double: the predicted value of y
     */
    inline double predict(double x) {
        if (Sxx == 0.0 || Sxy == 0.0) {
            get_results();
        }
        return a + b * x;
    }
};

#endif
