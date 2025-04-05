#include "../../../src/linalg/mat_2d.h"
#include "../../../third_party/catch.hpp"
#include <string>
#include <vector>

TEST_CASE("testing default constructor for Mat2d class") {
    Mat2d<int, 5, 5> mat;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            mat(i, j) = i + j;
        }
    }

    REQUIRE(mat(0, 0) == 0);
    REQUIRE(mat(1, 1) == 2);
    REQUIRE(mat(4, 3) == 7);
    REQUIRE(mat(4, 4) == 8);

    Mat2d<char, 5, 5> mat2({{'a', 'b', 'c', 'd', 'e'},
                            {'a', 'b', 'c', 'd', 'e'},
                            {'a', 'b', 'c', 'd', 'e'},
                            {'a', 'b', 'c', 'd', 'e'},
                            {'a', 'b', 'c', 'd', 'e'}});
    Mat1d<char, 5> check({'a', 'b', 'c', 'd', 'e'});
    bool ww = mat2(0) == check;
    REQUIRE(ww == true);
}

TEST_CASE("testing iterators for Mat2d class") {
    Mat2d<int, 5, 5> mat;
    for (int i = 0; i < mat.rows(); i++) {
        for (int j = 0; j < mat.cols(); j++) {
            mat(i, j) = i + j;
        }
    }

    Mat1d<int, 5> check({0, 1, 2, 3, 4});
    bool ww = mat(0) == check;
    Mat1d<int, 5> check2({1, 2, 3, 4, 5});
    bool ww2 = mat(1) == check2;
    REQUIRE(ww == true);
    REQUIRE(ww2 == true);
}

TEST_CASE("testing constructor with one value [1]") {
    Mat2d<int, 5, 5> mat(5);
    REQUIRE(mat.size() == 25);
    REQUIRE(mat.cols() == 5);
    REQUIRE(mat.rows() == 5);
    Mat1d<int, 5> check({5, 5, 5, 5, 5});
    REQUIRE(mat(0) == check);
}

TEST_CASE("testing operator << for Mat2d class") {
    Mat2d<char, 5, 5> mat('a');
    CHECK_NOTHROW(std::cout << mat << '\n');
    Mat2d<int, 5, 5> mat2(5);
    CHECK_NOTHROW(std::cout << mat2 << '\n');
    Mat2d<std::string, 10, 10> mat3("CSRT");
    CHECK_NOTHROW(std::cout << mat3 << '\n');
}

TEST_CASE("testing copy constructor for Mat2d class") {
    Mat2d<int, 3, 3> mat({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    Mat2d<int, 3, 3> mat2(mat);
    for (int i = 0; i < 3; i++) {
        REQUIRE(mat(i) == mat2(i));
    }
}

TEST_CASE("testing operator = for Mat2d class") {
    Mat2d<int, 10, 10> mat(10);
    Mat2d<int, 10, 10> mat2;
    mat2 = mat;
    for (int i = 0; i < 10; i++) {
        REQUIRE(mat(i) == mat2(i));
    }
}
