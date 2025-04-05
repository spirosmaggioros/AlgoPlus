#include "../../../src/machine_learning/regression/polynomial_regression/poly_reg.h"
#include "../../../third_party/json.hpp"
#include <fstream>
#include <iostream>

using nlohmann::json;

int main() {
    srand((unsigned)time(NULL));
    std::ifstream ifs("data.json");
    json jf = json::parse(ifs);
    std::vector<std::vector<double>> data = jf["data"];
    std::vector<double> X, Y;
    for (auto& x : data) {
        X.push_back(x[0]);
        Y.push_back(x[1]);
    }
    int64_t n = 3; // we can select the degree of the polynomial
    polynomial_regression a(X, Y, n);
    std::vector<double> b_coeffs = a.get_coeffs();
    json j;
    j["coeffs"] = b_coeffs;
    std::ofstream file("info.json");
    file << j;
    return 0;
}
