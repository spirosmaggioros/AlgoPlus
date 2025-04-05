#include "../../../../third_party/json.hpp"
#include <fstream>
#include <iostream>
#include "../../../../src/machine_learning/image/point_detection/laplacian_point_detect.h"

using namespace laplacian_detection;
using namespace std;
using namespace nlohmann;

int main() {
    std::ifstream ifs("data.json");
    json jf = json::parse(ifs);
    std::vector<std::vector<int32_t>> data = jf["img"];

    std::vector<std::vector<int32_t>> res = laplacian_detection::apply_point_detection(data, 220);

    json j;
    j["data"] = res;
    std::ofstream file("results.json");
    file << j;
}
