#include <iostream>
#include "../../../src/machine_learning/nn/perceptron.h"

int main() {
    std::vector<std::vector<double>> perceptron_data = {
        // Class -1 (left cluster)
        {0.1, 0.2, 0},
        {0.15, 0.25, 0},
        {0.05, 0.1, 0},
        {0.0, 0.1, 0},
        {0.2, 0.0, 0},
        {0.1, 0.15, 0},
        {0.05, 0.05, 0},
        {0.08, 0.12, 0},
        {0.12, 0.18, 0},
        {0.03, 0.15, 0},
        {0.18, 0.05, 0},
        {0.07, 0.03, 0},

        // Class +1 (right cluster)
        {1.2, 1.1, 1.0},
        {1.3, 1.2, 1.0},
        {1.1, 1.3, 1.0},
        {1.4, 1.0, 1.0},
        {1.25, 1.15, 1.0},
        {1.35, 1.25, 1.0},
        {1.15, 1.35, 1.0},
        {1.05, 1.1, 1.0},
        {1.1, 1.4, 1.0},
        {1.4, 1.3, 1.0},
        {1.3, 1.4, 1.0},
        {1.2, 1.25, 1.0},

        // Challenging near-boundary points
        {0.5, 0.5, 0},   // Should belong to -1 but close to boundary
        {0.6, 0.6, 1.0},    // Should belong to +1 but close to boundary
        {0.55, 0.55, 0}, // Another boundary case
        {0.65, 0.65, 1.0}   // Another boundary case
    };

    perceptron net(perceptron_data, 1, 15, 0.01);
    net.fit(1);

    std::vector<double> val_data = {0.15, 0.25};
    std::cout << net.predict(val_data) << '\n';
    val_data = {1.21, 1.24};
    std::cout << net.predict(val_data) << '\n';
}
