#include "../../../src/machine_learning/image/filters/average_filter.h"
#include "../../../third_party/catch.hpp"

using namespace avg_filter;

TEST_CASE("Testing average filter application") {
    std::vector<std::vector<int32_t>> image(50, std::vector<int32_t>(50, 42));

    auto resulted = apply_avg_filter(image);
    REQUIRE(resulted.size() != 0);
}

TEST_CASE("Testing average filter application with empty image") {
    std::vector<std::vector<int32_t>> image;
    CHECK_THROWS(apply_avg_filter(image));
}
