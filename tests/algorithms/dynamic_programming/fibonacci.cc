#include "../../../third_party/catch.hpp"
#include "../../../src/algorithms/dynamic_programming/fib.h"

TEST_CASE("testing fibonacci functions") {
    int64_t n1 = 2, ans1 = 1;
    int64_t n2 = 4, ans2 = 3;
    int64_t n3 = 29, ans3 = 514229;
    int64_t n4 = 15, ans4 = 610;
    REQUIRE(fibonacci(n1) == ans1);
    REQUIRE(fibonacci_dp(n1) == ans1);
    REQUIRE(fibonacci_bottom_up(n1) == ans1);
    REQUIRE(fibonacci_binet(n1) == ans1);

    REQUIRE(fibonacci(n2) == ans2);
    REQUIRE(fibonacci_dp(n2) == ans2);
    REQUIRE(fibonacci_bottom_up(n2) == ans2);
    REQUIRE(fibonacci_binet(n2) == ans2);

    REQUIRE(fibonacci(n3) == ans3);
    REQUIRE(fibonacci_dp(n3) == ans3);
    REQUIRE(fibonacci_bottom_up(n3) == ans3);
    REQUIRE(fibonacci_binet(n3) == ans3);

    REQUIRE(fibonacci(n4) == ans4);
    REQUIRE(fibonacci_dp(n4) == ans4);
    REQUIRE(fibonacci_bottom_up(n4) == ans4);
    REQUIRE(fibonacci_binet(n4) == ans4);
}