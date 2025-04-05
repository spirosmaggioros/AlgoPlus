#ifndef TREE_H
#define TREE_H

#ifdef ENABLE_TREE_VISUALIZATION
#include "../../visualization/tree_visual/tree_visualization.h"
#endif

#ifdef __cplusplus
#include <functional>
#include <iostream>
#include <memory>
#include <queue>
#endif

/**
 * @brief tree class
 *
 */
template <typename T> class tree {
  private:
    struct node {
        T info;
        std::shared_ptr<node> right;
        std::shared_ptr<node> left;
        node(T info) : info(info), left(nullptr), right(nullptr) {}
    };

    int64_t _size{};
    std::shared_ptr<node> root;

  public:
    /**
     * @brief constructor for tree class
     * @param v: the input vector of pairs(string and T) to avoid doing multiple
     * insertions
     */
    inline explicit tree(std::vector<std::pair<std::string, T>> v = {}) noexcept : root(nullptr) {
        if (!v.empty()) {
            for (std::pair<std::string, T>& x : v) {
                this->insert(x.first, x.second);
            }
        }
    }

    /**
     * @brief insert function
     * @param direction: string, directions for the insertion of value info
     * @param info: the value of the new node
     */
    inline void insert(std::string direction, T info) {
        std::shared_ptr<node> nn = std::make_shared<node>(info);
        if (!root) {
            _size++;
            root = nn;
            return;
        }
        int64_t i = 0;
        std::shared_ptr<node> head = root;
        while (head && i < direction.size() - 1) {
            if (direction[i] == 'r') {
                head = head->right;
            } else {
                head = head->left;
            }
            i++;
        }
        if (head && direction[i] == 'r') {
            head->right = nn;
        } else if (head && direction[i] == 'l') {
            head->left = nn;
        }
        _size++;
    }

    /**
     * @brief search function
     * @param key: the key to be searched
     * @return true: if the key exist in the tree
     * @return false: otherwise
     */
    inline bool search(T key) {
        std::queue<std::shared_ptr<node>> q;
        q.push(root);
        while (!q.empty()) {
            int64_t size = q.size();
            for (int64_t i = 0; i < size; ++i) {
                std::shared_ptr<node> current = q.front();
                q.pop();
                if (current->info == key) {
                    return true;
                }
                if (current->right) {
                    q.push(current->right);
                }
                if (current->left) {
                    q.push(current->left);
                }
            }
        }
        return false;
    }

    class Iterator;

    inline Iterator begin() {
        std::vector<T> ino = this->inorder();
        return Iterator(0, ino);
    }

    inline Iterator end() {
        std::vector<T> ino = this->inorder();
        return Iterator(ino.size(), ino);
    }

    /**
     * @brief inorder traversal
     * @return vector<T>: the inorder traversal of the tree
     */
    inline std::vector<T> inorder() {
        std::vector<T> ino;
        _inorder([&](std::shared_ptr<node> callbacked) { ino.push_back(callbacked->info); }, root);
        return ino;
    }

    /**
     * @brief postorder function
     * @return vector<T>: the postorder traversal of the tree
     */
    inline std::vector<T> postorder() {
        std::vector<T> path;
        _postorder([&](std::shared_ptr<node> callbacked) { path.push_back(callbacked->info); },
                   root);
        return path;
    }

    /**
     * @brief preorder function
     * @return vector<T>: the preorder traversal of the tree
     */
    inline std::vector<T> preorder() {
        std::vector<T> path;
        _preorder([&](std::shared_ptr<node> callbacked) { path.push_back(callbacked->info); },
                  root);
        return path;
    }

    /**
     * @brief level order function
     * @return vector<vector<T>>: the level order traversal of the tree
     */
    inline std::vector<std::vector<T>> level_order() {
        std::vector<std::vector<T>> path;
        std::queue<std::shared_ptr<node>> q;
        q.push(root);
        while (!q.empty()) {
            size_t size = q.size();
            std::vector<T> level;
            for (size_t i = 0; i < size; i++) {
                std::shared_ptr<node> current = q.front();
                q.pop();
                level.push_back(current->info);
                if (current->left) {
                    q.push(current->left);
                }
                if (current->right) {
                    q.push(current->right);
                }
            }
            path.push_back(level);
        }
        return path;
    }

    /**
     * @brief operator << for bst class
     */
    inline friend std::ostream& operator<<(std::ostream& out, tree<T>& t) {
        std::vector<T> order = t.inorder();
        for (int i = 0; i < order.size(); i++) {
            if (i != order.size() - 1) {
                out << order[i] << ", ";
            } else {
                out << order[i] << '\n';
            }
        }
        return out;
    }

    /**
     *@brief visualize function
     *@returns .dot file that can be previewed using graphviz in vscode.
     */
#ifdef TREE_VISUALIZATION_H
    inline void visualize() {
        std::string _generated = generate_visualization();
        tree_visualization::visualize(_generated);
    }
#endif

  private:
    void _inorder(std::function<void(std::shared_ptr<node>)> callback, std::shared_ptr<node> root) {
        if (root) {
            _inorder(callback, root->left);
            callback(root);
            _inorder(callback, root->right);
        }
    }

    void _preorder(std::function<void(std::shared_ptr<node>)> callback,
                   std::shared_ptr<node> root) {
        if (root) {
            callback(root);
            _preorder(callback, root->left);
            _preorder(callback, root->righ);
        }
    }

    void _postorder(std::function<void(std::shared_ptr<node>)> callback,
                    std::shared_ptr<node> root) {
        if (root) {
            _postorder(callback, root->right);
            callback(root);
            _postorder(callback, root->left);
        }
    }

    std::string generate_visualization() {
        std::string _generate = _inorder_gen(root);
        return _generate;
    }

    std::string _inorder_gen(std::shared_ptr<node> root) {
        std::string _s;
        if (std::is_same_v<T, char> || std::is_same_v<T, std::string>) {
            if (root->left) {
                _s += root->info;
                _s += "->";
                _s += root->left->info;
                _s += "\n";
                _s += _inorder_gen(root->left);
            }
            if (root->right) {
                _s += root->info;
                _s += "->";
                _s += root->right->info;
                _s += "\n";
                _s += _inorder_gen(root->right);
            }
        } else {
            if (root->left) {
                _s += std::to_string(root->info) + "->" + std::to_string(root->left->info) + "\n" +
                      _inorder_gen(root->left);
            }
            if (root->right) {
                _s += std::to_string(root->info) + "->" + std::to_string(root->right->info) + "\n" +
                      _inorder_gen(root->right);
            }
        }
        return _s;
    }
};

template <typename T> class tree<T>::Iterator {
  private:
    std::vector<T> elements;
    int64_t index;

  public:
    /**
     * @brief Construct a new Iterator object
     *
     * @param els vector<T> - the elements in inorder fashion
     */
    explicit Iterator(const int64_t& index, std::vector<T>& els) noexcept
        : index(index), elements(els) {}

    /**
     * @brief = operator for Iterator type
     *
     * @param index the current index
     * @return Iterator&
     */
    Iterator& operator=(int64_t index) {
        this->index = index;
        return *(this);
    }

    /**
     * @brief operator ++ for type Iterator
     *
     * @return Iterator&
     */
    Iterator& operator++() {
        if (this->index < elements.size()) {
            this->index++;
        }
        return *(this);
    }

    /**
     * @brief operator ++ for type Iterator
     *
     * @return Iterator
     */
    Iterator operator++(int) {
        Iterator it = *this;
        ++*(this);
        return it;
    }

    /**
     * @brief operator -- for type Iterator
     *
     * @return Iterator&
     */
    Iterator& operator--() {
        if (this->index > 0) {
            this->index--;
        }
        return *(this);
    }

    /**
     * @brief operator -- for type Iterator
     *
     * @return Iterator
     */
    Iterator operator--(int) {
        Iterator it = *this;
        --*(this);
        return it;
    }

    /**
     * @brief operator != for type Iterator
     *
     * @param it const Iterator
     * @return true if index == it.index
     * @return false otherwise
     */
    bool operator!=(const Iterator& it) { return index != it.index; }

    /**
     * @brief operator * for type Iterator
     *
     * @return T the value of the node
     */
    T operator*() { return elements[index]; }
};

#endif
