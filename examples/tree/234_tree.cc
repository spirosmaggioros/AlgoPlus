#include "../../src/classes/tree/234_tree.h"

#ifdef __cplusplus
#include <iostream>
#endif

int main() {
    ttf_tree<int> t;
    t.insert(30);
    t.insert(99);
    t.insert(70);
    t.insert(60);
    t.insert(40);
    t.insert(66);
    t.insert(50);
    t.insert(53);
    std::cout << t << '\n';
    t.insert(45);
    std::cout << t << '\n';
    t.insert(42);
    std::cout << t << '\n';
}
