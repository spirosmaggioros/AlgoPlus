#include "../../../../../../third_party/json.hpp"
#include <fstream>
#include <iostream>
#include "../../../../../../src/machine_learning/image/morphology/operations.h"

using namespace morphology_operations;
using namespace std;
using namespace nlohmann;

int main() {
    std::ifstream ifs("data.json");
    json jf = json::parse(ifs);
    std::vector<std::vector<int32_t>> data = jf["img"];
    for (auto& x : data) {
        for (auto& y : x) {
            if (y >= 150) {
                y = 255;
            } else {
                y = 0;
            }
        }
    }
    std::vector<std::vector<int32_t>> res = dilate(data);
    if (res.empty()) {
        cout << "empty" << '\n';
    }
    json j;
    j["data"] = res;
    std::ofstream file("results.json");
    file << j;
}