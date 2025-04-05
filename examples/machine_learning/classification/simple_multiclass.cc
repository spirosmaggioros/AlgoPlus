#include "../../../src/machine_learning/classification/multiclass_classification/simple_multiclass.h"
#include <iostream>
#include <vector>

int main() {
    std::vector<std::vector<double>> data = {{1.0, 2.0, 1.0, 0.0},
                                             {2.0, 1.0, 3.0, 1.0},
                                             {3.0, 2.0, 1.0, 2.0},
                                             {1.0, 3.0, 2.0, 0.0},
                                             {2.0, 3.0, 1.0, 1.0}};

    simple_multi_classification model(data, 3);
    model.fit();
}
