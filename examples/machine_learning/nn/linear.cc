#ifdef __cplusplus
#include "../../../src/machine_learning/nn/nn.h"
#endif

int main() {
    nn::Linear l1(10, 5, true);
    std::vector<double> in_tensor = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<double> out = l1.forward(in_tensor);

    for (auto& x : out) {
        std::cout << x << " ";
    }
    std::cout << '\n';
}
