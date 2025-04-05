#include "../../../src/machine_learning/image/filters/median_filter.h"
#include "../../../third_party/catch.hpp"

using namespace median_filter;

TEST_CASE("Testing median filter application") {
    std::vector<std::vector<int32_t>> image(50, std::vector<int32_t>(50, 42));

    auto resulted = apply_median_filter(image);
    REQUIRE(resulted.size() != 0);
}

TEST_CASE("Testing median filter application with empty image") {
    std::vector<std::vector<int32_t>> image;
    CHECK_THROWS(apply_median_filter(image));
}
