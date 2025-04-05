#include "../containers/parallelized_matrix.h"
#include "../../../third_party/catch.hpp"
#include "../../helpers/timer.h"

TEST_CASE("Testing addition for parallelized matrix") {
    std::vector<std::vector<int64_t>> v1 = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};

    std::vector<std::vector<int64_t>> v2 = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};

    std::vector<std::vector<int64_t>> result = parallel::add(v1, v2);
    std::vector<std::vector<int64_t>> check_result = {{2, 4, 6}, {2, 4, 6}, {2, 4, 6}};

    REQUIRE(check_result == result);
}

TEST_CASE("Testing addition for parallelized matrix [2]") {
    std::vector<std::vector<int64_t>> v1 = {{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}};

    std::vector<std::vector<int64_t>> v2 = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};

    std::vector<std::vector<int64_t>> result = parallel::add(v1, v2);
    std::vector<std::vector<int64_t>> check = {
        {3, 5, 7, 9, 11}, {3, 5, 7, 9, 11}, {3, 5, 7, 9, 11}};
    REQUIRE(check == result);
}

TEST_CASE("Testing addition for parallelized matrix [3]") {
    std::vector<std::vector<int64_t>> v1 = {
        {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}};

    std::vector<std::vector<int64_t>> v2 = {
        {1, 2, 3, 4, 5, 6, 7},
        {1, 2, 3, 4, 5, 6, 7},
        {1, 2, 3, 4, 5, 6, 7},
    };

    std::vector<std::vector<int64_t>> result = parallel::add(v1, v2);
    std::vector<std::vector<int64_t>> check = {
        {2, 3, 4, 5, 6, 7, 8},
        {2, 3, 4, 5, 6, 7, 8},
        {2, 3, 4, 5, 6, 7, 8},
    };

    REQUIRE(check == result);
}

TEST_CASE("Testing addition for a big parallelized matrix [1]") {
    std::vector<std::vector<int64_t>> v1(50, std::vector<int64_t>(50, 1)),
        v2(50, std::vector<int64_t>(50, 1));

    std::vector<std::vector<int64_t>> result = parallel::add(v1, v2);
    std::vector<std::vector<int64_t>> check(50, std::vector<int64_t>(50, 2));
    REQUIRE(check == result);
}

TEST_CASE("Testing addition for a big parallelized matrix [2]") {
    std::vector<std::vector<int64_t>> v1(500, std::vector<int64_t>(500, 1)),
        v2(500, std::vector<int64_t>(500, 1));
    std::vector<std::vector<int64_t>> result = parallel::add(v1, v2);
    std::vector<std::vector<int64_t>> check(500, std::vector<int64_t>(500, 2));
    REQUIRE(check == result);
}

TEST_CASE("Testing addition for a big parallelized matrix [3]") {
    std::vector<std::vector<int64_t>> v1(5000, std::vector<int64_t>(5000, 1)),
        v2(5000, std::vector<int64_t>(5000, 1));
    std::vector<std::vector<int64_t>> result = parallel::add(v1, v2);
    std::vector<std::vector<int64_t>> check(5000, std::vector<int64_t>(5000, 2));
    REQUIRE(check == result);
}

TEST_CASE("Testing addition for a big parallelized matrix [4]") {
    std::vector<std::vector<int64_t>> v1(10000, std::vector<int64_t>(10000, 1)),
        v2(10000, std::vector<int64_t>(10000, 1));
    std::vector<std::vector<int64_t>> result = parallel::add(v1, v2);
    std::vector<std::vector<int64_t>> check(10000, std::vector<int64_t>(10000, 2));
    REQUIRE(check == result);
}

TEST_CASE("Testing time of parallel vs serial for addition [1]") {
    std::vector<std::vector<int64_t>> v1(100000, std::vector<int64_t>(5000, 1)),
        v2(100000, std::vector<int64_t>(5000, 1));

    double start = clock();
    std::vector<std::vector<int64_t>> result = parallel::add(v1, v2);
    double end = clock();

    double fast = (end - start) / double(CLOCKS_PER_SEC) * 1000;
    std::cout << "computation with multithreading took " << fast << " ms" << '\n';

    start = clock();
    std::vector<std::vector<int64_t>> result2 = serial::add(v1, v2);
    end = clock();

    double slow = (end - start) / double(CLOCKS_PER_SEC) * 1000;
    std::cout << "serial computation took " << slow << " ms" << '\n';

    std::vector<std::vector<int64_t>> check(100000, std::vector<int64_t>(5000, 2));
    REQUIRE(result2 == check);
    REQUIRE(result == check);
}
