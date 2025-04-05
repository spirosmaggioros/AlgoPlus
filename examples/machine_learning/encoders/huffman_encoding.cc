#include "../../../src/machine_learning/image/encoders/huffman_encoding.h"

int main() {
    huffman h({"a", "b", "c"});
    h.create_tree();
    std::unordered_map<std::string, std::string> decoded = h.decode();

    for (auto& [x, y] : decoded) {
        std::cout << x << " -> " << y << '\n';
    }
}
