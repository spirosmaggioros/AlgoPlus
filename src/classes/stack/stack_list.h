#ifndef STACK_H
#define STACK_H

#ifdef __cplusplus
#include <iostream>
#include <memory>
#include <vector>
#endif

/**
 * @brief stack_list class
 */
template <typename T> class stack_list {
  private:
    /**
     * @brief struct for the node
     * @param val: the value of the node
     * @param next: pointer to the next node
     * @param prev: pointer to the previous node
     */
    struct node {
        T val;
        std::shared_ptr<node> next;
        std::shared_ptr<node> prev;
        node(T key) : val(key), next(nullptr), prev(nullptr) {}
    };

    std::shared_ptr<node> root;
    size_t _size{0};

  public:
    /**
     * @brief Construct a new stack list object
     *
     * @param v initializer vector
     */
    inline explicit stack_list(std::vector<T> v = {}) noexcept : root(nullptr) {
        if (!v.empty()) {
            for (T& x : v) {
                this->push(x);
            }
        }
    }

    /**
     * @brief Copy constructor for stack list class
     * @param s the stack we want to copy
     */
    inline explicit stack_list(const stack_list& s) : root(s.root), _size(s._size) {}

    /**
     * @brief operator = for stack list class
     * @param s the stack we want to copy
     * @return stack_list&
     */
    inline stack_list& operator=(const stack_list& s) {
        root = s.root;
        _size = s._size;
        return *this;
    }

    /**
     * @brief clear function
     */
    inline void clear() {
        root = nullptr;
        _size = 0;
    }

    /**
     * @brief size functon
     *
     * @return size_t the size of the stack
     */
    inline size_t size() { return _size; }

    /**
     * @brief push function
     *
     * @param key the key to be pushed
     */
    inline void push(T key) {
        std::shared_ptr<node> nn = std::make_shared<node>(key);
        if (!root) {
            root = nn;
            _size++;
            return;
        } else {
            root->next = nn;
            nn->prev = root;
            root = nn;
            _size++;
        }
    }

    /**
     * @brief top function
     *
     * @return T the top of the stack
     */
    inline T top() { return root->val; }

    /**
     * @brief pop function
     * removes the top of the stack
     */
    inline void pop() {
        root = root->prev;
        root->next = nullptr;
        _size--;
    }

    class Iterator;

    /**
     * @brief pointer to the top of the stack
     *
     * @return Iterator
     */
    inline Iterator begin() { return Iterator(root); }

    /**
     * @brief pointer to the end of the stack
     *
     * @return Iterator
     */
    inline Iterator end() { return Iterator(nullptr); }
};

/**
 * @brief Iterator class
 */
template <typename T> class stack_list<T>::Iterator {
  private:
    std::shared_ptr<node> curr_root;

  public:
    /**
     * @brief Construct a new Iterator object
     *
     * @param s stack_list pointer
     */
    explicit Iterator(const std::shared_ptr<node>& s) noexcept : curr_root(s) {}

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
            curr_root = curr_root->prev;
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
