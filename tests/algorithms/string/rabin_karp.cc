#include "../../../src/algorithms/string/rabin_karp.h"
#include "../../../third_party/catch.hpp"
#include <string>
#include <vector>

TEST_CASE("Testing Rabin Karp with no valid matches") {
    std::string text = "Hello world";
    std::string pattern = "abc";

    std::vector<size_t> result = rabin_karp(text, pattern);
    REQUIRE(result.empty());
}

TEST_CASE("Testing Rabin Karp with valid input") {
    std::string text = "Hello world";
    std::string pattern = "Hello";

    std::vector<size_t> result = rabin_karp(text, pattern);
    REQUIRE(result.size() == 1);
    REQUIRE(result[0] == 0);
}

TEST_CASE("Testing Rabin Karp with incorrect input") {
    std::string text = "abc";
    std::string pattern = "abcdefg";

    std::vector<size_t> result = rabin_karp(text, pattern);
    REQUIRE(result.empty());
}

TEST_CASE("Testing Rabin Karp with multiple occurrences") {
    std::string text = "abcabcabc";
    std::string pattern = "abc";

    std::vector<size_t> result = rabin_karp(text, pattern);
    REQUIRE(result.size() == 3);
    REQUIRE(result[0] == 0);
    REQUIRE(result[1] == 3);
    REQUIRE(result[2] == 6);
}

TEST_CASE("Testing Rabin Karp with empty text and non-empty pattern") {
    std::string text = "";
    std::string pattern = "abcdefg";

    std::vector<size_t> result = rabin_karp(text, pattern);
    REQUIRE(result.empty());
}

TEST_CASE("Testing Rabin Karp with non-empty text and empty pattern") {
    std::string text = "abcdefg";
    std::string pattern = "";

    std::vector<size_t> result = rabin_karp(text, pattern);
    REQUIRE(result.size() == text.length() + 1);
}

TEST_CASE("Testing Rabin Karp with both empty text and pattern") {
    std::string text = "";
    std::string pattern = "";

    std::vector<size_t> result = rabin_karp(text, pattern);
    REQUIRE(result.size() == 1);
    REQUIRE(result[0] == 0);
}

TEST_CASE("Testing Rabin Karp with pattern equals to the text") {
    std::string text = "Hello world";
    std::string pattern = "Hello world";

    std::vector<size_t> result = rabin_karp(text, pattern);
    REQUIRE(result.size() == 1);
    REQUIRE(result[0] == 0);
}

TEST_CASE("Stress testing Rabin Karp") {
    std::string text(100, 'a');
    std::string pattern(50, 'a');

    std::vector<size_t> result = rabin_karp(text, pattern);
    REQUIRE(result.size() == 100 - 50 + 1);
    for (size_t i = 0; i < result.size(); i++) {
        REQUIRE(result[i] == i);
    }
}

TEST_CASE("Testing Rabin Karp with multiple occurrences of duplicate characters") {
    std::string text = "aabcaaabcaaaab";
    std::string pattern = "aa";

    std::vector<size_t> result = rabin_karp(text, pattern);
    REQUIRE(result.size() == 6);
    REQUIRE(result[0] == 0);
    REQUIRE(result[1] == 4);
    REQUIRE(result[2] == 5);
    REQUIRE(result[3] == 9);
    REQUIRE(result[4] == 10);
    REQUIRE(result[5] == 11);
}
