#ifndef INTERVAL_TREE_H
#define INTERVAL_TREE_H

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
 *@brief interval tree class
 */
template <typename T> class interval_tree {
  public:
    /**
     * @brief Construct a new interval tree object
     *
     * @param v : vector<pair<T,T>> initializer
     */
    inline interval_tree(std::vector<std::pair<T, T>> v = {}) {
        if (!v.empty()) {
            for (auto& x : v) {
                this->insert({x.first, x.second});
            }
        }
    }

    /**
     * @brief Copy constructor for interval tree class
     *
     * @param i the tree we want to copy
     */
    inline explicit interval_tree(const interval_tree& i) : root(i.root), _size(i._size) {}

    /**
     * @brief operator = for interval tree class
     * @param i the tree we want to copy
     * @return interval_tree&
     */
    inline interval_tree& operator=(const interval_tree& i) {
        root = i.root;
        _size = i._size;
        return *this;
    }

    inline ~interval_tree() { root = nullptr; }

    /**
     * @brief clear function
     */
    inline void clear() {
        root = nullptr;
        _size = 0;
    }

    /**
     *@brief insert function.
     *@param p: interval to be inserted.
     */
    inline void insert(std::pair<T, T> p) {
        interval i = interval(p);
        root = _insert(root, i);
        _size++;
    }

    /**
     *@brief search function.
     *@returns true if an interval exist in the tree.
     */
    inline bool search(std::pair<T, T> p) {
        if (!root) {
            return false;
        }
        interval i = interval(p);
        if (this->overlap({root->i->low, root->i->high}, p)) {
            return true;
        }
        return _search(root, i);
    }

    /**
     *@brief remove function.
     *@param p: interval to be removed.
     */
    inline void remove(std::pair<T, T> p) {
        interval i = interval(p);
        root = _remove(root, i);
    }

    /**
     *@brief overlap function.
     *@param p1: first interval.
     *@param p2: second interval.
     *@returns true if p1 overlaps p2.
     */
    inline bool overlap(std::pair<T, T> p1, std::pair<T, T> p2) {
        interval i1 = interval(p1), i2 = interval(p2);
        return i1.high >= i2.low && i1.low <= i2.high;
    }

    class Iterator;

    /**
     * @brief pointer that points to begin
     *
     * @return Iterator
     */
    inline Iterator begin() {
        std::vector<std::pair<T, T>> ino = this->inorder();
        return Iterator(0, ino);
    }

    /**
     * @brief pointer that points to end
     *
     * @return Iterator
     */
    inline Iterator end() {
        std::vector<std::pair<T, T>> ino = this->inorder();
        return Iterator(ino.size(), ino);
    }

    /**
     * @brief size function
     *
     * @return size_t the size of the tree
     */
    inline size_t size() { return _size; }

    /**
     *@brief inorder function.
     *@returns vector<pair<T,T>>, the elements inorder.
     */
    inline std::vector<std::pair<T, T>> inorder() {
        std::vector<std::pair<T, T>> path;
        _inorder(
            [&](std::shared_ptr<node> callbacked) {
                path.push_back({callbacked->i->low, callbacked->i->high});
            },
            root);
        return path;
    }

    /**
     *@brief preorder function.
     *@returns vector<pair<T,T>>, the elements preorder.
     */
    inline std::vector<std::pair<T, T>> preorder() {
        std::vector<std::pair<T, T>> path;
        _preorder(
            [&](std::shared_ptr<node> callbacked) {
                path.push_back({callbacked->i->low, callbacked->i->high});
            },
            root);
        return path;
    }

    /**
     *@brief postorder function.
     *@returns vector<pair<T, T>>, the elements postorder.
     */
    inline std::vector<std::pair<T, T>> postorder() {
        std::vector<std::pair<T, T>> path;
        _postorder(
            [&](std::shared_ptr<node> callbacked) {
                path.push_back({callbacked->i->low, callbacked->i->high});
            },
            root);
        return path;
    }

    /**
     *@brief level order function.
     *@returns vector<pair<T, T>>, the level order traversal of the tree
     */
    inline std::vector<std::vector<std::pair<T, T>>> level_order() {
        std::vector<std::vector<std::pair<T, T>>> path;
        std::queue<std::shared_ptr<node>> q;
        q.push(root);
        while (!q.empty()) {
            size_t size = q.size();
            std::vector<std::pair<T, T>> level;
            for (size_t i = 0; i < size; i++) {
                std::shared_ptr<node> current = q.front();
                q.pop();
                level.push_back({current->i->low, current->i->high});
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

#ifdef TREE_VISUALIZATION_H
    inline void visualize() {
        std::string _generated = generate_visualization();
        tree_visualization::visualize(_generated);
    }
#endif

    /**
     * @brief operator << for interval tree class
     */
    inline friend std::ostream& operator<<(std::ostream& out, interval_tree<T>& t) {
        std::vector<std::vector<std::pair<T, T>>> order = t.level_order();
        for (std::vector<std::pair<T, T>>& x : order) {
            for (size_t i = 0; i < x.size(); i++) {
                if (i != x.size() - 1) {
                    out << '[' << x[i].first << "," << x[i].second << ']' << ", ";
                } else {
                    out << '[' << x[i].first << "," << x[i].second << ']' << '\n';
                }
            }
        }
        return out;
    }

  private:
    /**
     * @brief struct for the intervals
     * @param low: the min of the pair
     * @param high: the high of the pair
     */
    struct interval {
        T low;
        T high;
        interval(std::pair<T, T> p)
            : low(std::min(p.first, p.second)), high(std::max(p.first, p.second)) {}
    };

    /**
     * @brief struct for the node
     * @param i: the interval
     * @param max: max of the interval
     * @param right: pointer to the right
     * @param left: pointer to the left
     *
     */
    struct node {
        interval* i;
        int max;
        std::shared_ptr<node> right;
        std::shared_ptr<node> left;
        node(interval n) : i(new interval(n)), max(n.high), right(nullptr), left(nullptr) {}
    };

    std::shared_ptr<node> root;
    size_t _size{};

    std::shared_ptr<node> new_node(interval i) {
        std::shared_ptr<node> p = std::make_shared<node>(i);
        return p;
    }

    /**
     *@brief helper function for insertion
     */
    std::shared_ptr<node> _insert(std::shared_ptr<node> root, interval i) {
        if (!root) {
            return new_node(i);
        }
        T l = root->i->low;
        if (i.low < l) {
            root->left = _insert(root->left, i);
        } else {
            root->right = _insert(root->right, i);
        }
        if (root->max < i.high) {
            root->max = i.high;
        }
        return root;
    }

    /**
     *@brief helper function for search
     */
    bool _search(std::shared_ptr<node> root, interval i) {
        if (!root) {
            return false;
        }
        if (root->left && root->left->max >= i.low) {
            return _search(root->left, i);
        }
        return _search(root->right, i);
    }

    /**
     *@brief helper function for remove.
     */
    std::shared_ptr<node> _remove(std::shared_ptr<node> root, interval i) {
        if (!root) {
            return nullptr;
        }
        std::shared_ptr<node> p = root;
        while (p) {
            if (p->i->low > i.low) {
                p = p->left;
            } else if (p->i->low < i.low) {
                p = p->right;
            } else {
                if (!p->right && !p->left) {
                    _size--;
                    return nullptr;
                } else if (p->right && !p->left) {
                    std::shared_ptr<node> temp = root->right;
                    _size--;
                    return temp;
                } else if (p->left && !p->right) {
                    std::shared_ptr<node> temp = p->left;
                    _size--;
                    return temp;
                } else {
                    std::shared_ptr<node> temp = root;
                    while (temp && temp->left) {
                        temp = temp->left;
                    }
                    p->i = temp->i;
                    p->max = temp->max;
                    p->right = _remove(p->right, *temp->i);
                }
            }
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
                _s += '"';
                _s += root->i->low;
                _s += ',';
                _s += root->i->high;
                _s += '"';
                _s += "->";
                _s += '"';
                _s += root->left->i->low;
                _s += ',';
                _s += root->left->i->high;
                _s += '"';
                _s += "\n";
                _s += _inorder_gen(root->left);
            }
            if (root->right) {
                _s += '"';
                _s += root->i->low;
                _s += ',';
                _s += root->i->high;
                _s += '"';
                _s += "->";
                _s += '"';
                _s += root->right->i->low;
                _s += ',';
                _s += root->right->i->high;
                _s += '"';
                _s += "\n";
                _s += _inorder_gen(root->right);
            }
        } else {
            if (root->left) {
                _s += '"';
                _s += std::to_string(root->i->low);
                _s += ',';
                _s += std::to_string(root->i->high);
                _s += '"';
                _s += "->";
                _s += '"';
                _s += std::to_string(root->left->i->low);
                _s += ',';
                _s += std::to_string(root->left->i->high);
                _s += '"';
                _s += "\n";
                _s += _inorder_gen(root->left);
            }
            if (root->right) {
                _s += '"';
                _s += std::to_string(root->i->low);
                _s += ',';
                _s += std::to_string(root->i->high);
                _s += '"';
                _s += "->";
                _s += '"';
                _s += std::to_string(root->right->i->low);
                _s += ',';
                _s += std::to_string(root->right->i->high);
                _s += '"';
                _s += "\n";
                _s += _inorder_gen(root->right);
            }
        }
        return _s;
    }
};

/**
 * @brief Iterator class
 */
template <typename T> class interval_tree<T>::Iterator {
  private:
    std::vector<std::pair<T, T>> elements;
    int64_t index;

  public:
    /**
     * @brief Construct a new Iterator object
     *
     * @param els vector<pair<T,T>> - the elements in inorder fashion
     */
    explicit Iterator(const int64_t& index, std::vector<std::pair<T, T>>& els) noexcept
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
     * @return std::pair<T,T> the value of the node
     */
    std::pair<T, T> operator*() { return {elements[index].first, elements[index].second}; }
};

#endif
