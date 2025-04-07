#ifndef LINKED_LIST_H
#define LINKED_LIST_H

#ifdef ENABLE_LINKED_LIST_VISUALIZATION
#include "../../visualization/list_visual/linked_list_visualization.h"
#endif

#ifdef __cplusplus
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#endif

/**
 *@brief single linked list class
 */

template <typename T> class linked_list {
  public:
    /**
     *@brief linked_list class constructor
     *@param __elements: you can provide the constructor with a vector of elements
     *so you dont have to do multiple push backs yourself.
     */
    inline explicit linked_list(std::vector<T> _elements = {}) noexcept
        : root(nullptr), tail(nullptr) {
        if (!_elements.empty()) {
            for (T& x : _elements) {
                this->push_back(x);
            }
        }
    }

    /**
     * @brief copy constructor for the linked_list class
     * @param l the list we want to copy
     */
    inline explicit linked_list(const linked_list& l)
        : root(l.root), tail(l.tail), _size(l._size) {}

    /**
     * @brief operator = for linked list class
     * @param l the list we want to copy
     * @return linked_list&
     */
    inline linked_list& operator=(const linked_list& l) {
        root = l.root;
        tail = l.tail;
        _size = l._size;
        return *this;
    }

    /**
     *@brief empty function.
     *Returns true if the list is empty.
     */
    inline bool empty() { return root == nullptr; }

    /**
     *@brief size function.
     *Returns the size of the list.
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
     *@brief push_back function.
     *@param key: the key to be pushed back.
     */
    void push_back(T key);

    /**
     *@brief push_front function.
     *@param key: the key to be pushed in front.
     */
    void push_front(T key);

    /**
     *@brief erase function.
     *@param key: the key to be erased.
     */
    void erase(T key);

    /**
     *@brief search function.
     *@param key: the key to be searched.
     *@returns true if key exists in the list.
     */
    bool search(T key);

    /**
     *@brief elements function.
     *@returns vector<T>: the elements of the list.
     */
    std::vector<T> elements();

    /**
     *@brief reverse function.
     */
    void reverse();

    /**
     * @brief visualize function
     * returns a .dot file that can be previewd with graphviz plugin in vscode
     */
    void visualize();

    /**
     *@brief << operator for the linked_list class.
     */
    inline friend std::ostream& operator<<(std::ostream& out, linked_list<T>& l1) {
        out << '{';
        std::shared_ptr<node> head = l1.root;
        while (head) {
            out << head->val << ' ';
            head = head->next;
        }
        out << '}' << '\n';
        return out;
    }

  private:
    /**
     * @brief struct for the node
     * @param val: the value of the node
     * @param next: pointer to the next.
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

template <typename T> inline void linked_list<T>::push_back(T key) {
    std::shared_ptr<node> p = std::make_shared<node>(key);
    if (root == nullptr) {
        root = p;
    } else {
        tail->next = p;
    }
    tail = p;
    _size++;
}

template <typename T> inline void linked_list<T>::push_front(T key) {
    std::shared_ptr<node> p = std::make_shared<node>(key);
    p->next = root;
    if (tail == nullptr) {
        tail = root;
    }
    root = p;
    _size++;
}

template <typename T> inline void linked_list<T>::erase(T key) {
    if (empty()) {
        return;
    }
    std::shared_ptr<node> t = root;
    std::shared_ptr<node> to_be_removed = nullptr;
    while (t != tail && t->next->val != key) {
        t = t->next;
    }
    if (t == tail) {
        return;
    }
    to_be_removed = t->next;
    t->next = t->next->next;
    to_be_removed.reset();
    if (t->next == nullptr) {
        tail = t;
    }
    if (root == tail) {
        tail = nullptr;
    }
    _size--;
}

template <typename T> inline bool linked_list<T>::search(T key) {
    try {
        if (empty()) {
            return false;
        } else {
            std::shared_ptr<node> t = root;
            while (t != tail && t->val != key) {
                t = t->next;
            }
            if (t == tail || t == nullptr) {
                return false;
            }
            return true;
        }
    } catch (std::invalid_argument& e) {
        std::cerr << e.what() << '\n';
        return false;
    }
}

template <typename T> inline std::vector<T> linked_list<T>::elements() {
    std::vector<T> _elements;

    if (this->empty()) {
        return _elements;
    }
    std::shared_ptr<node> head = root;
    while (head) {
        _elements.push_back(head->val);
        head = head->next;
    }
    return _elements;
}

template <typename T> inline void linked_list<T>::reverse() {
    std::shared_ptr<node> current = root;
    std::shared_ptr<node> prev{nullptr}, next{nullptr};

    while (current != nullptr) {
        next = current->next;
        current->next = prev;
        prev = current;
        current = next;
    }
    root = prev;
}

template <typename T> inline std::string linked_list<T>::generate() {
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
        }
    }
    return gen;
}

#ifdef ENABLE_LINKED_LIST_VISUALIZATION
template <typename T> inline void linked_list<T>::visualize() {
    std::string generated = this->generate();
    linked_list_visualization::visualize(generated);
}
#endif

/**
 * @brief Iterator class
 */
template <typename T> class linked_list<T>::Iterator {
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
