#ifdef __cplusplus
#include "../../src/classes/heap/min_heap.h"
#include <iostream>
#endif

int main() {
    min_heap<char> m(5);
    m.insert('a');
    m.insert('z');
    m.insert('d');
    m.insert('e');
    std::cout << m.min() << '\n';
    m.remove(m.min());
    std::cout << m.min() << '\n';
}