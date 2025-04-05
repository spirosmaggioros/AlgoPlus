#ifndef CIRCULAR_LINKED_LIST_H
#define CIRCULAR_LINKED_LIST_H

#ifdef ENABLE_LIST_VISUALIZATION
#include "../../visualization/list_visual/linked_list_visualization.h"
#endif

#ifdef __cplusplus
#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>
#endif

/**
 * @brief circular linked list class
 */
template <typename T> class circular_linked_list {
  public:
    /**
     * @brief Construct a new circular linked list object
     *
     * @param __elements vector<T> so you dont have to do multiple insertions
     */
    inline explicit circular_linked_list(std::vector<T> _elements = {}) noexcept
        : root(nullptr), tail(nullptr) {
        if (!_elements.empty()) {
            for (T& x : _elements) {
                this->push_back(x);
            }
        }
    }

    /**
     * @brief copy constructor for the circular linked list class
     * @param c the list we want to copy
     */
    inline explicit circular_linked_list(const circular_linked_list& c)
        : root(c.root), tail(c.tail), _size(c._size) {}

    /**
     * @brief operator = for circular linked list class
     * @param c the list we want to copy
     * @return circular_linked_list&
     */
    inline circular_linked_list& operator=(const circular_linked_list& c) {
        root = c.root;
        tail = c.tail;
        _size = c._size;
        return *this;
    }

    /**
     * @brief empty function
     *
     * @return true if the list is empty
     * @return false otherwise
     */
    inline bool empty() { return root == nullptr; }

    /**
     * @brief size function
     *
     * @return size_t the size of the list
     */
    inline size_t size() { return _size; }

    class Iterator;

    /**
     * @brief pointer that points to begin
     *
     * @return Iterator
     */
    inline Iterator begin() { return Iterator(root); }

    /**
     * @brief pointer that points to end
     *
     * @return Iterator
     */
    inline Iterator end() { return Iterator(nullptr); }

    /**
     * @brief push_back function
     *
     * @param key the key to be inserted
     */
    void push_back(T key);

    /**
     * @brief push_front function
     *
     * @param key the key to be inserted
     */
    void push_front(T key);

    /**
     * @brief erase function
     *
     * @param key the key to be erased
     */
    void erase(T key);

    /**
     * @brief search function
     *
     * @param key the key to be searched
     * @return true if the key exist in the list
     * @return false otherwise
     */
    bool search(T key);

    /**
     * @brief elements function
     *
     * @return std::vector<T> the elements of the list
     */
    std::vector<T> elements();

    /**
     * @brief visualize function
     * returns a .dot file that can be previewd with graphviz plugin in vscode
     */
    void visualize();

    /**
     * @brief << operator for the circular list class
     */
    friend std::ostream& operator<<(std::ostream& out, circular_linked_list<T>& l1) {
        out << '{';
        std::shared_ptr<node> head = l1.root;
        do {
            out << head->val << ' ';
            head = head->next;
        } while (head != l1.root);
        out << '}' << '\n';
        return out;
    }

  private:
    /**
     * @brief struct for the node
     * @param val the value of the node
     * @param next smart pointer to the next node
     */
    struct node {
        T val;
        std::shared_ptr<node> next;
        node(T key = 0) : val(key), next(nullptr) {}
    };

    std::shared_ptr<node> root;
    std::shared_ptr<node> tail;
    size_t _size{0};

    std::string generate();
};

template <typename T> inline void circular_linked_list<T>::push_back(T key) {
    std::shared_ptr<node> p = std::make_shared<node>(key);
    if (root == nullptr) {
        root = p;
        tail = p;
    } else {
        tail->next = p;
        tail = p;
    }
    tail->next = root;
    _size++;
}

template <typename T> inline void circular_linked_list<T>::push_front(T key) {
    std::shared_ptr<node> p = std::make_shared<node>(key);
    if (root == nullptr) {
        root = p;
        tail = p;
    } else {
        p->next = root;
        root = p;
        tail->next = root;
    }
    _size++;
}

template <typename T> inline void circular_linked_list<T>::erase(T key) {
    if (empty()) {
        return;
    }

    std::shared_ptr<node> t = root;
    std::shared_ptr<node> to_be_removed = nullptr;

    do {
        if (t->next->val == key) {
            to_be_removed = t->next;
            t->next = t->next->next;
            to_be_removed.reset();
            _size--;

            if (t->next == root) {
                tail = t;
            }

            if (root == tail) {
                tail = nullptr;
            }

            return;
        }

        t = t->next;
    } while (t != root);
}

template <typename T> inline bool circular_linked_list<T>::search(T key) {
    try {
        if (empty()) {
            return false;
        } else {
            std::shared_ptr<node> t = root;
            do {
                if (t->val == key) {
                    return true;
                }
                t = t->next;
            } while (t != root);

            return false;
        }
    } catch (std::invalid_argument& e) {
        std::cerr << e.what() << '\n';
        return false;
    }
}

template <typename T> inline std::vector<T> circular_linked_list<T>::elements() {
    std::vector<T> _elements;

    if (this->empty()) {
        return _elements;
    }

    std::shared_ptr<node> head = root;
    do {
        _elements.push_back(head->val);
        head = head->next;
    } while (head != root);

    return _elements;
}

template <typename T> inline std::string circular_linked_list<T>::generate() {
    std::string gen;
    gen += "rankdir=LR;";
    gen += '\n';
    gen += "node [shape=record;]";
    gen += '\n';
    std::vector<T> els = this->elements();
    if (std::is_same_v<T, std::string> || std::is_same_v<T, char>) {
        for (auto& x : els) {
            gen += x;
            gen += " [label=<{ ";
            gen += x;
            gen += " | }>] ;";
            gen += '\n';
        }

        std::shared_ptr<node> curr = root;
        while (curr->next) {
            gen += curr->val;
            gen += ":ref -> ";
            gen += curr->next->val;
            gen += ":data [arrowhead=vee, arrowtail=dot, dir=both];";
            gen += '\n';
            curr = curr->next;
            if (curr == root) {
                break;
            }
        }
    } else {
        for (auto& x : els) {
            gen += std::to_string(x);
            gen += " [label=<{ ";
            gen += std::to_string(x);
            gen += " | }>] ;";
            gen += '\n';
        }

        std::shared_ptr<node> curr = root;
        while (curr->next) {
            gen += std::to_string(curr->val);
            gen += ":ref -> ";
            gen += std::to_string(curr->next->val);
            gen += ":data [arrowhead=vee, arrowtail=dot, dir=both];";
            gen += '\n';
            curr = curr->next;
            if (curr == root) {
                break;
            }
        }
    }
    return gen;
}

#ifdef ENABLE_LIST_VISUALIZATION
template <typename T> void circular_linked_list<T>::visualize() {
    std::string generated = this->generate();
    linked_list_visualization::visualize(generated);
}
#endif

/**
 * @brief Iterator class
 */
template <typename T> class circular_linked_list<T>::Iterator {
  private:
    std::shared_ptr<node> curr_root;

  public:
    /**
     * @brief Construct a new Iterator object
     *
     * @param l linked list type
     */
    explicit Iterator(const std::shared_ptr<node>& l) noexcept : curr_root(l) {}

    /**
     * @brief = operator for Iterator type
     *
     * @param current smart pointer of type node
     * @return Iterator&
     */
    Iterator& operator=(std::shared_ptr<node> current) {
        this->curr_root = current;
        return *(this);
    }

    /**
     * @brief operator ++ for type Iterator
     *
     * @return Iterator&
     */
    Iterator& operator++() {
        if (curr_root) {
            curr_root = curr_root->next;
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
     * @brief operator != for type Iterator
     *
     * @param it const Iterator
     * @return true if curr_root == it.curr_root
     * @return false otherwise
     */
    bool operator!=(const Iterator& it) { return curr_root != it.curr_root; }

    /**
     * @brief operator * for type Iterator
     *
     * @return T the value of the node
     */
    T operator*() { return curr_root->val; }
};
#endif
