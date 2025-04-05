#include "../../src/classes/tree/tree.h"
#include "../../src/visualization/tree_visual/tree_visualization.h"

#ifdef __cplusplus
#include <iostream>
#endif

int main() {
    tree<int> t;
    t.insert("", 10);
    t.insert("r", 15);
    t.insert("l", 20);
    t.insert("rr", 13);
    t.insert("rrr", 12);
    t.insert("rrl", 11);

    std::vector<int> ino = t.inorder();
    for (auto& x : ino) {
        std::cout << x << " ";
    }
    std::cout << '\n';
    t.visualize();
}
