#include "../../src/classes/list/doubly_linked_list.h"
#include "../../src/visualization/list_visual/linked_list_visualization.h"

#ifdef __cplusplus
#include <iostream>
#endif

int main() {
    doubly_linked_list<int> l1({5, 6}), l2;
    l1.push_back(1);
    l1.push_back(3);
    l1.push_back(4);
    std::cout << l1.size() << '\n';
    std::cout << l1 << '\n';
    if (l1.search(3)) {
        std::cout << "3 exists in the list" << '\n';
    }
    l1.visualize();
}
