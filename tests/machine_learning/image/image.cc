#include "../../../src/machine_learning/image/image.h"
#include "../../../third_party/catch.hpp"
#include <numeric>
#include <vector>

TEST_CASE("Testing default constructor of Image class") {
    std::vector<std::vector<int32_t>> img = {{1, 0, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0, 0}};
    Image i(img);
    REQUIRE(i.get_2d_array() == img);
    REQUIRE(i._height() == 4);
    REQUIRE(i._width() == 3);
}

TEST_CASE("Testing secondary constructor of Image class") {
    int height = 10, width = 10;
    Image i(height, width);
    REQUIRE(i._height() == height);
    REQUIRE(i._width() == width);
}

TEST_CASE("Testing getting point of a 2d image") {
    std::vector<std::vector<int32_t>> img = {{1, 0, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0, 0}};
    Image i(img);
    REQUIRE(i.get_point(0, 0) == 1);
    REQUIRE(i.get_point(0, 1) == 0);
}

TEST_CASE("Testing setting point at a 2d image") {
    std::vector<std::vector<int32_t>> img = {{1, 0, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0, 0}};
    Image i(img);

    REQUIRE(i.get_point(0, 0) == 1);
    i.set_point(0, 0, 5);
    REQUIRE(i.get_point(0, 0) == 5);
}

TEST_CASE("Testing adding to a point in 2d image") {
    std::vector<std::vector<int32_t>> img = {{1, 0, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0, 0}};
    Image i(img);

    REQUIRE(i.get_point(1, 1) == 0);
    i.add_2_point(1, 1, 4);
    REQUIRE(i.get_point(1, 1) == 4);
    i.add_2_point(1, 1, 5);
    REQUIRE(i.get_point(1, 1) == 9);
}

TEST_CASE("Testing binary checking in 2d image") {
    std::vector<std::vector<int32_t>> img = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    Image i(img);
    REQUIRE(i.binary() == true);
    i.set_point(0, 0, 1);
    REQUIRE(i.get_point(0, 0) == 1);
    REQUIRE(i.binary() == false);
}

TEST_CASE("Testing adding 2 2d images") {
    std::vector<std::vector<int32_t>> img = {{0, 0, 0}, {1, 1, 1}, {0, 0, 0}};
    std::vector<std::vector<int32_t>> img2 = {{0, 0, 0}, {1, 1, 1}, {0, 0, 0}};
    Image i1(img), i2(img2);
    i1 = i1.add(i2);
    std::vector<std::vector<int32_t>> check = {{0, 0, 0}, {2, 2, 2}, {0, 0, 0}};
    REQUIRE(i1.get_2d_array() == check);
}

TEST_CASE("Testing sub 2 2d images") {
    std::vector<std::vector<int32_t>> img = {{0, 0, 0}, {1, 1, 1}, {0, 0, 0}};
    std::vector<std::vector<int32_t>> img2 = {{0, 0, 0}, {1, 1, 1}, {0, 0, 0}};
    Image i1(img), i2(img2);
    i1 = i1.sub(i2);
    std::vector<std::vector<int32_t>> check = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    REQUIRE(i1.get_2d_array() == check);
}

TEST_CASE("Testing multiplying 2 2d images") {
    std::vector<std::vector<int32_t>> img = {{0, 0, 0}, {1, 1, 1}, {0, 0, 0}};
    std::vector<std::vector<int32_t>> img2 = {{0, 0, 0}, {1, 1, 1}, {0, 0, 0}};
    Image i1(img), i2(img2);
    i1 = i1.mul(i2);
    REQUIRE(i1.get_2d_array() == img2);
}

TEST_CASE("Testing applying 2d filter") {
    std::vector<std::vector<int32_t>> img = {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1},
                                             {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1},
                                             {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1},
                                             {1, 1, 1, 1}, {1, 1, 1, 1}};
    std::vector<std::vector<int32_t>> kernel = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    std::vector<std::vector<int32_t>> check(img.size(), std::vector<int32_t>(img[0].size(), 0));
    Image i(img);
    i = i.apply_filter2d(kernel);
    REQUIRE(i.get_2d_array() == check);

    std::vector<std::vector<float>> kernel2 = {
        {1.0 / 9, 1.0 / 9, 1.0 / 9}, {1.0 / 9, 1.0 / 9, 1.0 / 9}, {1.0 / 9, 1.0 / 9, 1.0 / 9}};

    CHECK_NOTHROW(i = i.apply_filter2d(kernel2));
}

TEST_CASE("Testing operator << for image class") {
    std::vector<std::vector<int32_t>> img = {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}};

    Image i(img);

    CHECK_NOTHROW(std::cout << i << '\n');

    std::vector<std::vector<int32_t>> img2(100, std::vector<int32_t>(100));

    for (int i = 0; i < 100; i++) {
        std::iota(img2[i].begin(), img2[i].end(), 1);
    }

    Image i2(img2);

    CHECK_NOTHROW(std::cout << i2 << '\n');
}
