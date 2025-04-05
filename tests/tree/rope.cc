#define ENABLE_TREE_VISUALIZATION
#include "../../src/classes/tree/rope.h"
#include "../../third_party/catch.hpp"

TEST_CASE("Testing rope constructor [1]") {
    rope r("hello world");
    REQUIRE(r.height() == 1);
    REQUIRE(r.inorder() == "hello world");
}

TEST_CASE("Testing rope constructor [2]") {
    rope r("import numpy as np", 2);
    REQUIRE(r.height() == 1);
    REQUIRE(r.inorder() == "import numpy as np");

    rope rr("import numpy as np", 3);
    REQUIRE(r.height() <= 2);
    REQUIRE(r.inorder() == "import numpy as np");
    // CHECK_NOTHROW(rr.visualize());
}
