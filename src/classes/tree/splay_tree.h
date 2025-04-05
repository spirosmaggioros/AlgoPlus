#ifndef SPLAY_TREE_H
#define SPLAY_TREE_H

#ifdef ENABLE_TREE_VISUALIZATION
#include "../../visualization/tree_visual/tree_visualization.h"
#endif

#ifdef __cplusplus
#include <functional>
#include <iostream>
#include <memory>
#include <queue>
#include <vector>
#endif

/**
 * @brief splay tree class
 */
template <typename T> class splay_tree {
  private:
    /**
     * @brief stuct for the node
     * @param info: the value of the node
     * @param right: pointer to the right
     * @param left: pointer to the left
     */
    struct node {
        T info;
        std::shared_ptr<node> right;
        std::shared_ptr<node> left;
        node(T key = 0) : info(key), right(nullptr), left(nullptr) {}
    };
    std::shared_ptr<node> root;
    size_t _size{0};

  public:
    /**
     * @brief Construct a new splay tree object
     *
     * @param v vector<T> initializer vector
     */
    inline explicit splay_tree(std::vector<T> v = {}) noexcept : root(nullptr) {
        if (!v.empty()) {
            for (T& x : v) {
                this->insert(x);
            }
        }
    }

    /**
     * @brief Copy constructor for splay tree class
     * @param s the tree we want to copy
     */
    inline explicit splay_tree(const splay_tree& s) : root(s.root), _size(s._size) {}

    /**
     * @brief operator = for splay tree class
     * @param s the tree we want to copy
     * @return splay_tree&
     */
    inline splay_tree& operator=(const splay_tree& s) {
        root = s.root;
        _size = s._size;
        return *this;
    }

    inline ~splay_tree() { root = nullptr; }

    /**
     * @brief clear function
     */
    inline void clear() {
        root = nullptr;
        _size = 0;
    }

    /**
     * @brief insert function
     *
     * @param key the key to be inserted
     */
    inline void insert(T key) {
        root = _insert(root, key);
        _size++;
    }

    /**
     * @brief remove function
     *
     * @param key the key to be removed
     */
    inline void remove(T key) {
        root = _remove(root, key);
        _size--;
    }

    /**
     * @brief search function
     *
     * @param key
     * @return true if key exists in the tree
     * @return false otherwise
     */
    inline bool search(T key) {
        std::shared_ptr<node> ans = splay(root, key);
        return (ans && ans->info == key);
    }

    /**
     * @brief size function
     *
     * @return size_t the size of the tree
     */
    inline size_t size() { return _size; }

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
     *@brief inorder function.
     *@returns vector<T>, the elements inorder.
     */
    inline std::vector<T> inorder() {
        std::vector<T> path;
        _inorder([&](std::shared_ptr<node> callbacked) { path.push_back(callbacked->info); }, root);
        return path;
    }

    /**
     *@brief preorder function.
     *@returns vector<T>, the elements preorder.
     */
    inline std::vector<T> preorder() {
        std::vector<T> path;
        _preorder([&](std::shared_ptr<node> callbacked) { path.push_back(callbacked->info); },
                  root);
        return path;
    }

    /**
     *@brief postorder function.
     *@returns vector<T>, the elements postorder.
     */
    inline std::vector<T> postorder() {
        std::vector<T> path;
        _postorder([&](std::shared_ptr<node> callbacked) { path.push_back(callbacked->info); },
                   root);
        return path;
    }

    /**
     * @brief level order function
     * @return vector<vector<T>>, the level order traversal of the tree
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
     *@brief visualize function
     *@returns .dot file that can be previewed using graphviz in vscode.
     */
#ifdef TREE_VISUALIZATION_H
    inline void visualize() {
        std::string _generated = generate_visualization();
        tree_visualization::visualize(_generated);
    }
#endif

    /**
     * @brief operator << for splay tree class
     */
    inline friend std::ostream& operator<<(std::ostream& out, splay_tree<T>& t) {
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

  private:
    std::shared_ptr<node> rrotate(std::shared_ptr<node> _node) {
        std::shared_ptr<node> y = _node->left;
        _node->left = y->right;
        y->right = _node;
        return y;
    }

    std::shared_ptr<node> lrotate(std::shared_ptr<node> _node) {
        std::shared_ptr<node> y = _node->right;
        _node->right = y->left;
        y->left = _node;
        return y;
    }

    std::shared_ptr<node> splay(std::shared_ptr<node> _node, T key) {
        if (!_node || _node->info == key) {
            return _node;
        }
        if (_node->info > key) {
            if (_node->left == NULL)
                return _node;
            if (_node->left->info > key) {
                _node->left->left = splay(_node->left->left, key);
                _node = rrotate(_node);
            } else if (_node->left->info < key) {
                _node->left->right = splay(_node->left->right, key);
                if (_node->left->right != NULL)
                    _node->left = lrotate(_node->left);
            }
            return (_node->left == NULL) ? _node : rrotate(_node);
        } else {
            if (_node->right == NULL)
                return _node;
            if (_node->right->info > key) {
                _node->right->left = splay(_node->right->left, key);
                if (_node->right->left != NULL)
                    _node->right = rrotate(_node->right);
            } else if (_node->right->info < key) {
                _node->right->right = splay(_node->right->right, key);
                _node = lrotate(_node);
            }
            return (_node->right == NULL) ? _node : lrotate(_node);
        }
    }

    std::shared_ptr<node> _insert(std::shared_ptr<node> root, T key) {
        if (!root) {
            std::shared_ptr<node> nn = std::make_shared<node>(key);
            return nn;
        }
        root = splay(root, key);
        if (root->info == key) {
            return root;
        }

        std::shared_ptr<node> nn = std::make_shared<node>(key);
        if (root->info > key) {
            nn->right = root;
            nn->left = root->left;
            root->left = nullptr;
        } else {
            nn->left = root;
            nn->right = root->right;
            root->right = nullptr;
        }
        return nn;
    }

    std::shared_ptr<node> _remove(std::shared_ptr<node> root, T key) {
        if (!root) {
            return nullptr;
        }
        root = splay(root, key);
        if (key != root->info) {
            return root;
        }

        std::shared_ptr<node> temp;
        temp = root;
        if (!root->left) {
            root = root->right;
        } else {
            root = splay(root->left, key);
            root->right = temp->right;
        }
        return root;
    }

    void _inorder(std::function<void(std::shared_ptr<node>)> callback, std::shared_ptr<node> root) {
        if (root) {
            _inorder(callback, root->left);
            callback(root);
            _inorder(callback, root->right);
        }
    }

    void _postorder(std::function<void(std::shared_ptr<node>)> callback,
                    std::shared_ptr<node> root) {
        if (root) {
            _postorder(callback, root->left);
            _postorder(callback, root->right);
            callback(root);
        }
    }

    void _preorder(std::function<void(std::shared_ptr<node>)> callback,
                   std::shared_ptr<node> root) {
        if (root) {
            callback(root);
            _preorder(callback, root->left);
            _preorder(callback, root->right);
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

/**
 * @brief Iterator class
 */
template <typename T> class splay_tree<T>::Iterator {
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
