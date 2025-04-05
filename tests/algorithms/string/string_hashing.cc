#include "../../../src/algorithms/string/string_hashing.h"
#include "../../../third_party/catch.hpp"
#include <string>
#include <vector>

TEST_CASE("Testing string hasher") {
    string_hasher hash;
    hash.insert("hello");
    hash.insert("we");
    hash.insert("are");
    hash.insert("csrt");
    REQUIRE(hash.search("csrt") == true);
    REQUIRE(hash.search("abcd") == false);
    REQUIRE(hash.search("are") == true);
}

TEST_CASE("Testing string hasher size function") {
    string_hasher hash;
    std::string str = "abc";
    for (int i = 0; i < str.size(); i++) {
        std::string tmp = "";
        for (int j = i; j < str.size(); j++) {
            tmp += str[j];
            hash.insert(tmp);
        }
    }

    REQUIRE(hash.size() == 6);
}
