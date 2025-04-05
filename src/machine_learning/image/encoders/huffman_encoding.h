#ifndef HUFFMAN_ENCODING_H
#define HUFFMAN_ENCODING_H

#ifdef __cplusplus
#include <cassert>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
#endif

/**
 * @brief class for huffman coding
 */
class huffman {
  private:
    struct node {
        double weight;
        std::string ID;
        std::shared_ptr<node> left;
        std::shared_ptr<node> right;
        node(std::string ID, double weight)
            : ID(ID), weight(weight), left(nullptr), right(nullptr) {}
    };
    double _size{};
    int64_t MAX_DEPTH;
    std::shared_ptr<node> root;
    std::unordered_map<char, double> appearances;
    std::priority_queue<std::pair<double, std::shared_ptr<node>>,
                        std::vector<std::pair<double, std::shared_ptr<node>>>,
                        std::greater<std::pair<double, std::shared_ptr<node>>>>
        _weights;
    std::unordered_map<char, double> map_weights;

  public:
    /**
     * @brief constructor for huffman class
     * @param v: vector of strings(i.e. the text)
     * @param MAX_DEPTH: maximum depth(number of bits) that we want to
     * have(default = 10)
     */
    explicit huffman(std::vector<std::string> v = {}, int64_t MAX_DEPTH = 10)
        : root(nullptr), MAX_DEPTH(MAX_DEPTH) {
        for (std::string& x : v) {
            for (char& y : x) {
                appearances[y]++;
                _size++;
            }
        }
    }

    /**
     * @brief create_tree function
     * creates the tree in recursive fashion, from the leafs up to the root
     */
    inline void create_tree() {
        compute_weights();
        while (_weights.size() != 1) {
            auto first_node = _weights.top();
            _weights.pop();
            auto second_node = _weights.top();
            _weights.pop();
            std::shared_ptr<node> nn =
                std::make_shared<node>(first_node.second->ID + second_node.second->ID,
                                       first_node.first + second_node.first);
            nn->right = second_node.second;
            nn->left = first_node.second;
            _weights.push({nn->weight, nn});
        }
        root = _weights.top().second;
    }

    /**
     * @brief decode function
     * @return unordered_map<string, string>: the resulted encoding
     */
    inline std::unordered_map<std::string, std::string> decode() {
        std::vector<int> v(MAX_DEPTH);
        int top = 0;
        std::unordered_map<std::string, std::string> decoded;
        _decode(root, v, top, decoded);
        return decoded;
    }

  private:
    void compute_weights() {
        for (auto& x : appearances) {
            std::string curr = "";
            curr += x.first;
            std::shared_ptr<node> nn = std::make_shared<node>(curr, x.second / _size);
            _weights.push({x.second / _size, nn});
            map_weights[x.first] = x.second / _size;
        }
    }

    void _decode(std::shared_ptr<node> root, std::vector<int> arr, int top,
                 std::unordered_map<std::string, std::string>& decoded) {
        if (root->left) {
            arr[top] = 0;
            _decode(root->left, arr, top + 1, decoded);
        }
        if (root->right) {
            arr[top] = 1;
            _decode(root->right, arr, top + 1, decoded);
        }
        if (!root->left && !root->right) {
            std::string ans = "";
            for (int i = 0; i < top; i++) {
                ans += std::to_string(arr[i]);
            }
            decoded[root->ID] = ans;
        }
    }
};

#endif
