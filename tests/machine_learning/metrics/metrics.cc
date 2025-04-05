#include "../../../src/machine_learning/metrics/metrics.h"
#include "../../../third_party/catch.hpp"

using namespace metrics;
using namespace metrics::losses;

TEST_CASE("Testing accuracy score") {
    std::vector<double> y{1, 0, 1, 1, 0, 1, 0, 0, 1, 1};
    std::vector<double> y_pred{1, 0, 1, 0, 0, 1, 0, 0, 1, 1};

    REQUIRE(accuracy_score(y, y_pred) == Approx(0.9).epsilon(1e-6));
}

TEST_CASE("Testing recall") {
    std::vector<double> y{1, 0, 1, 1, 0, 1, 0, 0, 1, 1};
    std::vector<double> y_pred{1, 0, 1, 0, 0, 1, 0, 0, 1, 1};

    REQUIRE(recall(y, y_pred) == Approx(0.83333333).epsilon(1e-6));
}

TEST_CASE("Testing precision") {
    std::vector<double> y{1, 0, 1, 1, 0, 1, 0, 0, 1, 1};
    std::vector<double> y_pred{1, 0, 1, 0, 0, 1, 0, 0, 1, 1};

    REQUIRE(precision(y, y_pred) == Approx(1.0).epsilon(1e-6));
}

TEST_CASE("Testing F1 score") {
    std::vector<double> y{1, 0, 1, 1, 0, 1, 0, 0, 1, 1};
    std::vector<double> y_pred{1, 0, 1, 0, 0, 1, 0, 0, 1, 1};

    REQUIRE(f1_score(y, y_pred) == Approx(0.90909090).epsilon(1e-6));
}

TEST_CASE("Testing mean squared error") {

    std::vector<double> v1{1.23, 4.25, 4.4, 1.231, 5.567};
    std::vector<double> v2{4.56, 4.123, 1.234, 6.432, 5.555};

    REQUIRE(mean_squared_error(v1, v2) == Approx(9.6358263926).epsilon(1e-6));
}

TEST_CASE("Testing mean absolute error") {
    std::vector<double> v1{1.23, 4.25, 4.4, 1.231, 5.567};
    std::vector<double> v2{4.56, 4.123, 1.234, 6.432, 5.555};

    REQUIRE(mean_absolute_error(v1, v2) == Approx(2.36720).epsilon(1e-6));
}

TEST_CASE("Testing binary crossentropy loss") {
    std::vector<double> v1{0., 0., 0., 0., 1., 0., 0., 0., 0., 1.};
    std::vector<double> v2{1.8055,  0.9193,  -0.2527, 1.0489,  0.5396,
                           -1.2046, -0.9479, 0.8274,  -0.0548, -0.1902};

    REQUIRE(binary_crossentropy_loss(v1, v2) == Approx(0.8834657559).epsilon(1e-6));
}

TEST_CASE("Testing euclidean distance") {
    std::vector<double> x = {1.2, 4.3, 2.2, 1.1};
    std::vector<double> y = {0.2, 4.4, 1.1, 2.2};

    REQUIRE(metrics::euclidean_distance(x, y) == Approx(1.8520259177452134));
}

TEST_CASE("Testing manhattan distance") {
    std::vector<double> x = {1.2, 4.3, 2.2, 1.1};
    std::vector<double> y = {0.2, 4.4, 1.1, 2.2};

    REQUIRE(metrics::manhattan_distance(x, y) == 3.3000000000000007);
}

TEST_CASE("Testing minkowski distance") {
    std::vector<double> x = {1, 0, 0};
    std::vector<double> y = {0, 1, 0};

    REQUIRE(metrics::minkowski_distance(x, y, 1) == 2.0);
    REQUIRE(metrics::minkowski_distance(x, y, 2) == Approx(1.4142135623730951));
    REQUIRE(metrics::minkowski_distance(x, y, 3) == Approx(1.2599210498948732));
}
