#include "../../../../../third_party/json.hpp"
#include <fstream>
#include <iostream>
#include "../../../../../src/machine_learning/image/filters/gaussian_filter.h"

using namespace gaussian_filter;
using namespace std;
using namespace nlohmann;

int main() {
    std::ifstream ifs("data.json");
    json jf = json::parse(ifs);
    std::vector<std::vector<int32_t>> data = jf["img"];
    std::vector<std::vector<int32_t>> res = apply_gaussian_filter(data);
    if (res.empty()) {
        cout << "empty" << '\n';
    }
    json j;
    j["data"] = res;
    std::ofstream file("results.json");
    file << j;
}
