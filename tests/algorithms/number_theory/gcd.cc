#include "../../../src/algorithms/number_theory/gcd.h"
#include "../../../third_party/catch.hpp"

TEST_CASE("testing binary gcd") {
    int64_t a = 100, b = 4;
    REQUIRE(binary_gcd(a, b) == 4);
    REQUIRE(binary_gcd(b, a) == 4);
    REQUIRE(euclidean_gcd(a, b) == 4);
    a = 100;
    b = 400;
    REQUIRE(binary_gcd(a, b) == 100);
    REQUIRE(euclidean_gcd(a, b) == 100);
    a = 90;
    b = 85430;
    REQUIRE(binary_gcd(a, b) == 10);
    REQUIRE(euclidean_gcd(a, b) == 10);
}