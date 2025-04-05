#include "../../src/classes/tree/rope.h"
#include <iostream>
#include "../../src/visualization/tree_visual/tree_visualization.h"

using namespace std;

int main() {
    rope r("hello world", 3);

    cout << r.inorder() << '\n';

    r.visualize();
}
