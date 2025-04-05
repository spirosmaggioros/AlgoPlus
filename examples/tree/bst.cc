#include "../../src/classes/tree/bst.h"
#include <iostream>
#include "../../src/visualization/tree_visual/tree_visualization.h"

int main() {
    bst<int> b({1, 3, 5});
    b.insert(1);
    b.insert(10);
    b.insert(-2);
    std::vector<int> v = b.inorder();
    for (auto& x : v) {
        std::cout << x << " ";
    }
    std::cout << '\n';

    // now you can visualize trees in Algo+

    bst<char> b2;
    b2.insert('a');
    b2.insert('t');
    b2.insert('f');
    b2.insert('w');
    b2.insert('p');
    b2.insert('o');
    b2.insert('m');

    b2.visualize();
}
