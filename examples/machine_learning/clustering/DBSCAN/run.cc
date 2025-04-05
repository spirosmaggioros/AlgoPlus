#include "../../../../third_party/json.hpp"
#include <fstream>
#include <random>
#include "../../../../src/machine_learning/clustering/DBSCAN/dbscan.h"

using nlohmann::json;

int main(int argc, char* argv[]) {
    srand((unsigned)time(NULL));
    std::ifstream ifs("data.json");
    json jf = json::parse(ifs);
    std::vector<std::pair<double, double>> data = jf["data"];

    DBSCAN a(data, 4, 3);
    std::map<std::pair<double, double>, int64_t> clusters = a.get_clusters();
    std::vector<std::pair<double, double>> noise = a.get_noise();
    json j;
    j["noise"] = noise;
    for (auto& x : clusters) {
        std::cout << "[" << x.first.first << " " << x.first.second << "]" << ": " << x.second
                  << '\n';
    }
    j["centroids"] = clusters;
    std::ofstream file("info.json");
    file << j;
}
