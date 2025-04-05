#ifndef POLY_REG_H
#define POLY_REG_H

#ifdef __cplusplus
#include <cmath>
#include <iostream>
#include <vector>
#endif

class polynomial_regression {
  private:
    std::vector<double> X;
    std::vector<double> Y;
    int64_t n;

  public:
    explicit polynomial_regression(std::vector<double> X, std::vector<double> Y, int64_t n) noexcept
        : X(X), Y(Y), n(n) {}

    inline std::vector<double> get_coeffs() {
        std::vector<std::vector<double>> k = calculate_matrix(this->X, this->n);
        std::vector<double> l = calculate_vector(this->X, this->Y, this->n);
        std::vector<double> b_coeffs = solve_linear_system(k, l);
        return b_coeffs;
    }

    inline std::vector<std::vector<double>> create_matrix(int64_t rows, int64_t cols) {
        return std::vector<std::vector<double>>(rows, std::vector<double>(cols, 0.0));
    }

    inline std::vector<std::vector<double>> calculate_matrix(const std::vector<double>& x,
                                                             int64_t n) {
        int64_t m = x.size();
        auto matrix = create_matrix(n + 1, n + 1);
        for (int64_t i = 0; i <= n; ++i) {
            for (int64_t j = 0; j <= n; ++j) {
                for (int64_t k = 0; k < m; ++k) {
                    matrix[i][j] += pow(x[k], i + j);
                }
            }
        }
        return matrix;
    }

    inline std::vector<double> calculate_vector(const std::vector<double>& x,
                                                const std::vector<double>& y, int64_t n) {
        int64_t m = x.size();
        std::vector<double> vector(n + 1, 0.0);
        for (int64_t i = 0; i <= n; ++i) {
            for (int64_t k = 0; k < m; ++k) {
                vector[i] += y[k] * pow(x[k], i);
            }
        }
        return vector;
    }

    // Gaussian elimination
    inline std::vector<double> solve_linear_system(std::vector<std::vector<double>> A,
                                                   std::vector<double> b) {
        int64_t n = A.size();
        for (int64_t i = 0; i < n; ++i) {

            int64_t maxRow = i;
            for (int64_t k = i + 1; k < n; ++k) {
                if (std::abs(A[k][i]) > std::abs(A[maxRow][i])) {
                    maxRow = k;
                }
            }

            std::swap(A[maxRow], A[i]);
            std::swap(b[maxRow], b[i]);

            for (int64_t k = i + 1; k < n; ++k) {
                double c = -A[k][i] / A[i][i];
                for (int64_t j = i; j < n; ++j) {
                    if (i == j) {
                        A[k][j] = 0;
                    } else {
                        A[k][j] += c * A[i][j];
                    }
                }
                b[k] += c * b[i];
            }
        }

        std::vector<double> x(n);
        for (int64_t i = n - 1; i >= 0; --i) {
            x[i] = b[i] / A[i][i];
            for (int64_t k = i - 1; k >= 0; --k) {
                b[k] -= A[k][i] * x[i];
            }
        }
        return x;
    }

    /**
     * @brief predict function
     * @param x: the value of x which we want to predict y
     * @return double: the predicted  value of y
     */
    inline double predict(double x) {
        std::vector<double> coeffs = get_coeffs();
        double y_pred = 0.0;

        for (int64_t i = 0; i < int(coeffs.size()); i++) {
            y_pred += coeffs[i] * std::pow(x, i);
        }

        return y_pred;
    }
};

#endif
