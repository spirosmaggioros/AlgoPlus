#include "../../../../src/machine_learning/clustering/kmeans/kmeans.h"
#include "../../../../third_party/json.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    int CLUSTERS = atoi(argv[1]);
    srand((unsigned)time(NULL));
    std::ifstream ifs("data.json");
    json jf = json::parse(ifs);
    std::vector<std::vector<double>> data = jf["data"];

    json j;
    j["data"] = data;
    kmeans a(data, CLUSTERS);
    std::pair<std::vector<std::vector<double>>, std::map<std::vector<double>, int64_t>> ans =
        a.fit();
    j["centroids"] = ans.first;
    j["assignments"] = ans.second;
    std::ofstream file("info.json");
    file << j;
}
