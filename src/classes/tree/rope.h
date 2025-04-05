#ifndef ROPE_H
#define ROPE_H

#ifdef TREE_VISUALIZATION_H
#include "../../visualization/tree_visual/tree_visualization.h"
#endif

#ifdef __cplusplus
#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>
#endif

/**
 * @brief rope's(chord's) data structure class
 * **************************WARNING*****************************
 * This is a class under current development, this implementation
 * doesn't comply with the implementations that you might've seen online.
 * Please read the code carefully if you plan to use this class
 * **************************************************************
 * This class needs a lot of optimizations, like the way it performs splitting.
 * Splitting right now keeps BFS'ing on all the nodes, but we always know that
 * only the leaf nodes will split. Also, at insertion we have to search for
 * already existing subtrees and point the new node there instead of creating
 * duplicates.
 */
class rope {
    struct node {
        int64_t len;
        bool is_leaf;
        std::shared_ptr<node> left;
        std::shared_ptr<node> right;
        std::string data;

        explicit node(int64_t len, std::shared_ptr<node> left, std::shared_ptr<node> right) {
            this->len = len;
            this->is_leaf = false;
            this->left = left;
            this->right = right;
        }

        explicit node(std::string s) {
            this->len = s.size();
            this->is_leaf = true;
            this->left = nullptr;
            this->right = nullptr;
            this->data = s;
        }
    };

    void _inorder(std::function<void(std::shared_ptr<node>)> callback, std::shared_ptr<node> root) {
        if (root) {
            _inorder(callback, root->left);
            callback(root);
            _inorder(callback, root->right);
        }
    }

    std::vector<std::shared_ptr<node>> ropes;
    std::shared_ptr<node> root;
    int64_t _height;

  public:
    explicit rope(std::string s, int64_t splits = 2) : root(nullptr), _height(-1) {
        this->create(s, splits);
    }

    void create(std::string s, int64_t splits = 2);

    std::string inorder();

    int64_t height() { return this->_height; }

#ifdef TREE_VISUALIZATION_H
    void visualize() {
        std::string _generated = generate_visualization();
        tree_visualization::visualize(_generated);
    }
#endif

  private:
    std::string generate_visualization() {
        std::string _generate = _inorder_gen(root);
        return _generate;
    }

    std::shared_ptr<node> split_tree(std::shared_ptr<node> root, int64_t& splits) {
        std::queue<std::shared_ptr<node>> q;
        q.push(root);
        while (splits) {
            q = std::queue<std::shared_ptr<node>>{};
            q.push(root);
            int64_t count_height = -1;
            while (!q.empty()) {
                int64_t size = q.size();
                for (int64_t i = 0; i < size; i++) {
                    std::shared_ptr<node> curr = q.front();
                    q.pop();
                    if (splits <= 0) {
                        q = std::queue<std::shared_ptr<node>>{};
                        break;
                    }
                    if (curr->is_leaf == true) {
                        std::string s1 = (curr->data).substr(0, int(curr->len) / 2 + 1);
                        std::string s2 = (curr->data)
                                             .substr(int(curr->len) / 2 + 1,
                                                     int(curr->len) - int(curr->len) / 2);
                        std::shared_ptr<node> split1 = std::make_shared<node>(s1);
                        std::shared_ptr<node> split2 = std::make_shared<node>(s2);

                        std::shared_ptr<node> block_node =
                            std::make_shared<node>(curr->len, split1, split2);
                        curr->data.clear();
                        curr->is_leaf = block_node->is_leaf;
                        curr->len = block_node->len;
                        curr->left = block_node->left;
                        curr->right = block_node->right;

                        splits--;
                    }

                    if (curr->left != nullptr) {
                        q.push(curr->left);
                    }
                    if (curr->right != nullptr) {
                        q.push(curr->right);
                    }
                }
                count_height++;
                _height = std::max(_height, count_height);
            }
        }
        return root;
    }

    std::string _inorder_gen(std::shared_ptr<node> root) {
        std::string _s;
        if (root->left) {
            if (root->is_leaf) {
                _s += '"' + std::to_string(root->len) + ',' + root->data + '"';
            } else {
                _s += '"' + std::to_string(root->len) + '"';
            }
            _s += "->";
            if (root->left->is_leaf) {
                _s += '"' + std::to_string(root->left->len) + ',' + root->left->data + '"';
            } else {
                _s += '"' + std::to_string(root->left->len) + '"';
            }
            _s += "\n";
            _s += _inorder_gen(root->left);
        }
        if (root->right) {
            if (root->is_leaf) {
                _s += '"' + std::to_string(root->len) + ',' + root->data + '"';
            } else {
                _s += '"' + std::to_string(root->len) + '"';
            }
            _s += "->";
            if (root->right->is_leaf) {
                _s += '"' + std::to_string(root->right->len) + ',' + root->right->data + '"';
            } else {
                _s += '"' + std::to_string(root->right->len) + '"';
            }
            _s += "\n";
            _s += _inorder_gen(root->right);
        }
        return _s;
    }
};

void rope::create(std::string s, int64_t splits) {
    assert(s.size() >= splits);

    std::string s1 = s.substr(0, (int(s.size()) / 2) + 1);
    std::string s2 = s.substr(int(s.size()) / 2 + 1, int(s.size()) - int(s.size() / 2));
    std::shared_ptr<node> split1 = std::make_shared<node>(s1);
    std::shared_ptr<node> split2 = std::make_shared<node>(s2);

    std::shared_ptr<node> root_block = std::make_shared<node>(s.size(), split1, split2);
    root = root_block;
    splits--;
    root = split_tree(root, splits);
}

std::string rope::inorder() {
    std::string path;
    _inorder(
        [&](std::shared_ptr<node> callbacked) {
            if (callbacked->is_leaf == true) {
                path += callbacked->data;
            }
        },
        root);
    return path;
}

#endif
