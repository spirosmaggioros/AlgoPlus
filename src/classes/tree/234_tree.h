#ifndef TWOTHREEFOUR_TREE_H
#define TWOTHREEFOUR_TREE_H

#ifdef __cplusplus
#include <algorithm>
#include <iostream>
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>
#endif

/**
 * @brief 234-tree class
 */
template <typename T> class ttf_tree {
  private:
    /**
     * @brief node structure of 234-class
     * @param keys the elements of the node
     * @param numChildren the number of children that this node has(indicates the
     * type of the node as well)
     * @param index the current index of the children(0-3)
     * @param children pointers to the 4 total childrens of the node
     */
    typedef struct node {
        std::vector<T> keys;
        std::vector<std::shared_ptr<node>> children;
        int numChildren;
        int index{};
        node(std::vector<T> keys, std::vector<std::shared_ptr<node>> children, int numChildren)
            : keys(keys), children(children), numChildren(numChildren) {}
    } node;

    std::shared_ptr<node> root;
    std::unordered_map<std::shared_ptr<node>, std::shared_ptr<node>> parent;

  public:
    /**
     * @brief default constructor of 234-tree class
     * @param elements: by default empty, if you want to pass elements all in one
     * to the tree
     */
    explicit ttf_tree(const std::vector<T>& elements = {}) {
        if (!elements.empty()) {
            for (const T& x : elements) {
                this->insert(x);
            }
        }
    }

    /**
     * @brief search function
     * @param key the element we want to search
     * @return true if key exists in the tree
     * @return false otherwise
     */
    bool search(const T& key) const;

    /**
     * @brief insert function
     * @param key thet key we want to insert
     */
    void insert(const T& key);

    /**
     * @brief level_order function
     * @return vector<vector<vector<T> > > the level order of the 234-tree in a 3d
     * array
     */
    std::vector<std::vector<std::vector<T>>> level_order() const;

    /**
     * @brief operator << for ttf_tree class
     */
    friend std::ostream& operator<<(std::ostream& out, ttf_tree<T>& t) {
        std::vector<std::vector<std::vector<T>>> level_ordered = t.level_order();
        for (auto& x : level_ordered) {
            for (auto& y : x) {
                out << '[';
                for (int i = 0; i < y.size(); i++) {
                    if (i != y.size() - 1) {
                        out << y[i] << ", ";
                    } else {
                        out << y[i] << "], ";
                    }
                }
            }
            out << '\n';
        }
        return out;
    }
};

template <typename T> inline void ttf_tree<T>::insert(const T& key) {
    std::vector<std::shared_ptr<node>> null_children(4, nullptr);
    if (root == nullptr) {
        std::vector<int> keys = {key};
        root = std::make_shared<node>(keys, null_children, 2);
        root->index = 0;
        parent[root] = nullptr;
        return;
    } else {
        std::shared_ptr<node> head = root;
        while (head != nullptr) {
            if (head->numChildren == 2) { // case for 2-node
                if (head->children[0] == nullptr && head->children[1] == nullptr) {
                    head->keys.push_back(key);
                    std::sort(head->keys.begin(), head->keys.end());
                    head->numChildren++;
                    return;
                } else if (key == head->keys[0]) {
                    return;
                } else if (key < head->keys[0]) {
                    head = head->children[0];
                } else if (key > head->keys[0]) {
                    head = head->children[1];
                }
            } else if (head->numChildren == 3) { // case for 3-node
                if (head->children[0] == nullptr && head->children[1] == nullptr &&
                    head->children[2] == nullptr) {
                    head->keys.push_back(key);
                    std::sort(head->keys.begin(), head->keys.end());
                    head->numChildren++;
                    return;
                } else if (key == head->keys[0] || key == head->keys[1]) {
                    return;
                } else if (key < head->keys[0]) {
                    head = head->children[0];
                } else if (key > head->keys[0] && key < head->keys[1]) {
                    head = head->children[1];
                } else if (key > head->keys[1]) {
                    head = head->children[2];
                }
            } else { // case for 4-node
                std::shared_ptr<node> parent_node = parent[head];
                if (parent_node == nullptr) {
                    if (head->numChildren == 2 || head->numChildren == 3) {
                        std::shared_ptr<node> saved_root = head;
                        std::vector<T> passed_keys = {saved_root->keys[1]};
                        root = std::make_shared<node>(passed_keys, null_children, 2);
                        passed_keys = {saved_root->keys[0]};
                        root->children[0] = std::make_shared<node>(passed_keys, null_children, 2);
                        root->children[0]->index = 0;
                        parent[root->children[0]] = root;
                        passed_keys = {saved_root->keys[2]};
                        root->children[1] = std::make_shared<node>(passed_keys, null_children, 2);
                        root->children[1]->index = 1;
                        parent[root->children[1]] = root;
                        head = root;
                    } else {
                        std::shared_ptr<node> saved_root = head;
                        std::vector<T> passed_keys = {saved_root->keys[1]};
                        root = std::make_shared<node>(passed_keys, null_children, 2);
                        passed_keys = {saved_root->keys[0]};
                        root->children[0] = std::make_shared<node>(passed_keys, null_children, 2);
                        parent[root->children[0]] = root;
                        passed_keys = {saved_root->keys[2]};
                        root->children[1] = std::make_shared<node>(passed_keys, null_children, 2);
                        parent[root->children[1]] = root;
                        root->children[0]->children[0] = saved_root->children[0];
                        parent[root->children[0]->children[0]] = root->children[0];
                        root->children[0]->children[1] = saved_root->children[1];
                        parent[root->children[0]->children[1]] = root->children[0];
                        root->children[1]->children[0] = saved_root->children[2];
                        parent[root->children[1]->children[0]] = root->children[1];
                        root->children[1]->children[1] = saved_root->children[3];
                        parent[root->children[1]->children[1]] = root->children[1];
                        head = root;
                    }
                } else if (parent_node->numChildren == 2) {
                    int curr_index = head->index;
                    std::vector<T> saved_keys = head->keys;
                    parent_node->keys.push_back(saved_keys[1]);
                    parent_node->numChildren++;
                    std::sort(parent_node->keys.begin(), parent_node->keys.end());
                    if (curr_index == 0) {
                        parent_node->children[curr_index + 2] =
                            parent_node->children[curr_index + 1];
                    }
                    std::vector<T> passed_keys = {saved_keys[0]};
                    parent_node->children[curr_index] =
                        std::make_shared<node>(passed_keys, null_children, 2);
                    parent[parent_node->children[curr_index]] = parent_node;
                    parent_node->children[curr_index]->index = curr_index;
                    passed_keys = {saved_keys[2]};
                    parent_node->children[curr_index + 1] =
                        std::make_shared<node>(passed_keys, null_children, 2);
                    parent[parent_node->children[curr_index + 1]] = parent_node;
                    parent_node->children[curr_index + 1]->index = curr_index + 1;
                    head = parent_node;
                } else if (parent_node->numChildren == 3) {
                    int curr_index = head->index;
                    std::vector<T> saved_keys = head->keys;
                    parent_node->keys.push_back(saved_keys[1]);
                    parent_node->numChildren++;
                    std::sort(parent_node->keys.begin(), parent_node->keys.end());
                    if (curr_index == 0) {
                        parent_node->children[curr_index + 3] =
                            parent_node->children[curr_index + 2];
                        parent_node->children[curr_index + 2] =
                            parent_node->children[curr_index + 1];
                    } else if (curr_index == 1) {
                        parent_node->children[curr_index + 2] =
                            parent_node->children[curr_index + 1];
                    }
                    std::vector<T> passed_keys = {saved_keys[0]};
                    parent_node->children[curr_index] =
                        std::make_shared<node>(passed_keys, null_children, 2);
                    parent_node->children[curr_index]->index = curr_index;
                    parent[parent_node->children[curr_index]] = parent_node;
                    passed_keys = {saved_keys[2]};
                    parent_node->children[curr_index + 1] =
                        std::make_shared<node>(passed_keys, null_children, 2);
                    parent[parent_node->children[curr_index + 1]] = parent_node;
                    parent_node->children[curr_index + 1]->index = curr_index + 1;
                    if (key == parent_node->keys[0] || key == parent_node->keys[1] ||
                        key == parent_node->keys[2]) {
                        return;
                    } else if (key < parent_node->keys[0]) {
                        head = parent_node->children[0];
                    } else if (key > parent_node->keys[0] && key < parent_node->keys[1]) {
                        head = parent_node->children[1];
                    } else if (key > parent_node->keys[1] && key < parent_node->keys[2]) {
                        head = parent_node->children[2];
                    } else if (key > parent_node->keys[2]) {
                        head = parent_node->children[3];
                    }
                }
            }
        }
    }
}

template <typename T> inline bool ttf_tree<T>::search(const T& key) const {
    std::shared_ptr<node> head = root;
    while (head != nullptr) {
        if (head->numChildren == 2) { // case for 2-node
            if (key == head->keys[0] || key == head->keys[1]) {
                return true;
            } else if (key < head->keys[0]) {
                head = head->children[0];
            } else if (key > head->keys[1]) {
                head = head->children[1];
            }
        } else if (head->numChildren == 3) { // case for 3-node
            if (key == head->keys[0] || key == head->keys[1]) {
                return true;
            } else if (key < head->keys[0]) {
                head = head->children[0];
            } else if (key > head->keys[0] && key < head->keys[1]) {
                head = head->children[1];
            } else if (key > head->keys[1]) {
                head = head->children[2];
            }
        } else { // case for 4-node
            if (key == head->keys[0] || key == head->keys[1] || key == head->keys[2]) {
                return true;
            } else if (key < head->keys[0]) {
                head = head->children[0];
            } else if (key > head->keys[0] && key < head->keys[1]) {
                head = head->children[1];
            } else if (key > head->keys[1] && key < head->keys[2]) {
                head = head->children[2];
            } else if (key > head->keys[2]) {
                head = head->children[3];
            }
        }
    }
    return false;
}

template <typename T>
inline std::vector<std::vector<std::vector<T>>> ttf_tree<T>::level_order() const {
    std::vector<std::vector<std::vector<T>>> level_ordered;
    std::queue<std::shared_ptr<node>> q;
    q.push(root);
    while (!q.empty()) {
        std::vector<std::vector<T>> line;
        size_t size = q.size();
        for (size_t i = 0; i < size; i++) {
            auto current = q.front();
            q.pop();
            line.push_back(current->keys);
            for (auto x : current->children) {
                if (x != nullptr) {
                    q.push(x);
                }
            }
        }
        level_ordered.push_back(line);
    }
    return level_ordered;
}

#endif
