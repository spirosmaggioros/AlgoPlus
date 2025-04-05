#include "../../../src/machine_learning/regression/linear_regression/lin_reg.h"
#include "../../../third_party/json.hpp"
#include <fstream>
#include <iostream>
#include <string>

using nlohmann::json;

int main() {
    srand((unsigned)time(NULL));
    std::ifstream ifs("data.json");
    json jf = json::parse(ifs);
    std::vector<std::vector<double>> data = jf["data"];
    linear_regression lin_reg(data);
    std::pair<double, double> a_b = lin_reg.get_results();
    std::cout << a_b.first << " " << a_b.second << '\n';
    json j;
    j["a"] = a_b.first;
    j["b"] = a_b.second;
    std::ofstream file("info.json");
    file << j;
}
