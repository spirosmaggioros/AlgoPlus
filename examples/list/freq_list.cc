#include "../../src/classes/list/frequency_list.h"
#include "../../src/visualization/list_visual/linked_list_visualization.h"

#ifdef __cplusplus
#include <iostream>
#endif

int main() {
    frequency_list<int> f({1, 2, 3, 2, 4, 5, 3, 3, 4});

    for (auto it = f.begin(); it != f.end(); it++) {
        std::cout << *(it) << '\n';
    }
    // this should preview a list like { 3|3, 2|2, 4|2, 1|1, 5|1}
    f.visualize();
}
