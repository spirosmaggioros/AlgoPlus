#include "../../../src/algorithms/searching/interpolation_search.h"
#include "../../../third_party/catch.hpp"

TEST_CASE("Searching for existing elements, interpolation_search") {
    std::vector<int> data = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};

    REQUIRE(interpolation_search(data.begin(), data.end(), 15) == 7);
    REQUIRE(interpolation_search(data.begin(), data.end(), 7) == 3);
    REQUIRE(interpolation_search(data.begin(), data.end(), 1) == 0);
    REQUIRE(interpolation_search(data.begin(), data.end(), 19) == 9);
}

TEST_CASE("Searching for a non-existing element, interpolation_search") {
    std::vector<int> data = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};

    REQUIRE(interpolation_search(data.begin(), data.end(), 0) == -1);
    REQUIRE(interpolation_search(data.begin(), data.end(), 22) == -1);
}

TEST_CASE("Empty data array, interpolation_search") {
    std::vector<int> empty_data;

    REQUIRE(interpolation_search(empty_data.begin(), empty_data.end(), 1) == -1);
}

TEST_CASE("Single element array, interpolation_search") {
    std::vector<int> single_element = {7};

    REQUIRE(interpolation_search(single_element.begin(), single_element.end(), 7) == 0);
    REQUIRE(interpolation_search(single_element.begin(), single_element.end(), 1) == -1);
}