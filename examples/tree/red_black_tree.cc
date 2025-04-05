#include "../../src/classes/tree/red_black_tree.h"
#include "../../src/visualization/tree_visual/tree_visualization.h"

#ifdef __cplusplus
#include <iostream>
#endif

int main() {
    red_black_tree<char> rb({'a', 'b', 'c'});
    rb.insert('d');
    rb.insert('t');
    rb.remove('b');
    // Will print the inorder traversal.
    std::cout << rb << '\n';

    // Will return the preorder traversal.
    std::vector<char> v = rb.preorder();
    for (int i = 0; i < v.size(); i++)
        std::cout << v[i] << " ";
    std::cout << std::endl;

    // example for visualize.
    red_black_tree<int> rb2({1, 2, 3, 4, 5, 6, 7, 8, 9});
    rb2.visualize();
    return 0;
}
