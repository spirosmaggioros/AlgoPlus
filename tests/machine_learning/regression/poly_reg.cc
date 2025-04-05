#include "../../../src/machine_learning/regression/polynomial_regression/poly_reg.h"
#include "../../../third_party/catch.hpp"
#include <string>

TEST_CASE("Testing polynomial regression [1]") {
    std::vector<std::vector<double>> data = {{1, 45000},  {2, 50000},  {3, 60000}, {4, 80000},
                                             {5, 110000}, {6, 150000}, {7, 200000}};
    std::vector<double> X, Y;
    for (auto& x : data) {
        X.push_back(x[0]);
        Y.push_back(x[1]);
    }
    int64_t n = 3; // we can select the degree of the polynomial
    polynomial_regression a(X, Y, n);
    std::vector<double> b_coeffs = a.get_coeffs();
    std::vector<double> check = {48571.428571428056, -6388.888888888403, 3035.7142857141635,
                                 138.8888888888979};
    for (int i = 0; i < check.size(); i++) {
        REQUIRE(int(b_coeffs[i]) == int(check[i]));
    }

    REQUIRE_NOTHROW(a.predict(4));
}
