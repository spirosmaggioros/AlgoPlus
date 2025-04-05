#include "../../src/classes/tree/interval_tree.h"
#include "../../src/visualization/tree_visual/tree_visualization.h"

#ifdef __cplusplus
#include <iostream>
#endif

int main() {
    interval_tree<int> i;
    i.insert({20, 36});
    i.insert({3, 41});
    i.insert({29, 99});
    i.insert({0, 1});
    i.insert({10, 15});

    // you can visualize the tree
    i.visualize();

    // you can get the elements in any order
    std::vector<std::pair<int, int>> __preorder = i.preorder();
    std::vector<std::pair<int, int>> __inorder = i.inorder();
    // ....
}
