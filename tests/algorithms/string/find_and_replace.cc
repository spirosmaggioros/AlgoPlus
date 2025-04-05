#include "../../../src/algorithms/string/find_and_replace.h"
#include "../../../third_party/catch.hpp"
#include <string>

TEST_CASE("testing find_and_replace") {
    std::string str = "abcllllabc";
    find_and_replace(str, "abc", "def");
    REQUIRE(str == "deflllldef");
}

TEST_CASE("testing find_and_replace_regex") {
    std::string str = "abcbca";
    find_and_replace_regex(str, "(a.*?b.*?c)", "def");
    REQUIRE(str == "defbca");
}

TEST_CASE("testing removing substring for find_and_replace") {
    std::string str = "abcllllabc";
    find_and_replace(str, "abc", "");
    REQUIRE(str == "llll");
}

TEST_CASE("testing removing substring for find_and_replace_regex") {
    std::string str = "abcbca";
    find_and_replace_regex(str, "(a.*?b.*?c)", "");
    REQUIRE(str == "bca");
}