#include "../../../src/algorithms/string/kmp.h"
#include "../../../third_party/catch.hpp"
#include <string>

TEST_CASE("testing kmp") {
    std::string a = "ababcabcababababd", b = "ababd";
    REQUIRE(kmp(b, a) == true);
    REQUIRE(kmp("elo", a) == false);
}