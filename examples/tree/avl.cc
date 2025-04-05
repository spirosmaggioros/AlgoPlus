#include "../../src/classes/tree/avl_tree.h"
#include "../../src/visualization/tree_visual/tree_visualization.h"

#ifdef __cplusplus
#include <iostream>
#endif

int main() {
    avl_tree<int> a;
    avl_tree<std::vector<int>> b;
    b.insert({4, 5, 6});
    b.insert({1, 3, 4});
    b.insert({5, 6, 7});
    std::vector<std::vector<int>> r = b.inorder();
    for (auto& x : r) {
        for (auto& y : x) {
            std::cout << y << ' ';
        }
        std::cout << '\n';
    }
    std::cout << '\n';
    a.insert(1);
    a.insert(-10);
    a.insert(10);
    std::vector<int> v = a.inorder();
    for (auto& x : v) {
        std::cout << x << " ";
    }
    std::cout << '\n';

    // you can now visualize with Algo+
    avl_tree<int> b2;
    b2.insert(10);
    b2.insert(15);
    b2.insert(-10);
    b2.insert(-5);
    b2.insert(-8);

    b2.visualize();

    avl_tree<int> testing_avl;
    testing_avl.insert(10);
    testing_avl.visualize();
    testing_avl.insert(5);
    testing_avl.visualize();
    testing_avl.insert(-5);
    testing_avl.visualize();
    testing_avl.insert(2);
    testing_avl.visualize();
}
