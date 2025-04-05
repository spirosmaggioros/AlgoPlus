#include "../../../third_party/catch.hpp"
#include "../../../src/machine_learning/activation/activation_functions.h"

TEST_CASE("Testing ReLU activation function") {
    double x = 0.01;

    REQUIRE(activation::ReLU(x) == 0.01);
    x = -1;
    REQUIRE(activation::ReLU(x) == 0);
    x = 10.0;
    REQUIRE(activation::ReLU(x) == 10.0);
}

TEST_CASE("Testing LeakyReLU activation function") {
    double x = 0.01;

    REQUIRE(activation::LeakyReLU(x, 0.01) == 0.01);
    x = -1.0;
    REQUIRE(activation::LeakyReLU(x, 0.01) == -0.01);
    REQUIRE(activation::LeakyReLU(x, 0.0001) == -0.0001);
}

TEST_CASE("Testing softmax activation function") {
    std::vector<double> logits = {1.3, 5.1, 2.2, 0.7, 1.1};
    std::vector<double> probs = activation::softmax(logits);
    std::vector<double> check = {0.0201904647, 0.902537689, 0.049, 0.01, 0.0165305544};

    for (int i = 0; i < int(check.size()); i++) {
        REQUIRE(check[i] == Approx(probs[i]).epsilon(1e-1));
    }
}
