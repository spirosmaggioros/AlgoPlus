#ifndef QUEUE_H
#define QUEUE_H

#ifdef __cplusplus
#include <iostream>
#include <memory>
#include <vector>
#endif

/**
 * @brief dequeue list class
 */
template <typename T> class dequeue_list {
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
    std::shared_ptr<node> tail;
    size_t _size{0};

  public:
    /**
     * @brief Construct a new dequeue list object
     *
     * @param v initializer vector
     */
    inline explicit dequeue_list(std::vector<T> v = {}) noexcept : root(nullptr), tail(nullptr) {
        if (!v.empty()) {
            for (T& x : v) {
                this->push_back(x);
            }
        }
    }

    /**
     * @brief Copy constructor for dequeue list class
     *
     * @param q the dequeue we want to copy
     */
    inline explicit dequeue_list(const dequeue_list& q)
        : root(q.root), tail(q.tail), _size(q._size) {}

    /**
     * @brief operator = for dequeue list class
     * @param q the dequeue we want to copy
     * @return dequeue_list&
     */
    inline dequeue_list& operator=(const dequeue_list& q) {
        root = q.root;
        tail = q.tail;
        _size = q._size;
        return *this;
    }

    /**
     * @brief clear function
     */
    inline void clear() {
        root = nullptr;
        tail = nullptr;
        _size = 0;
    }

    /**
     * @brief size functon
     *
     * @return size_t the size of the stack
     */
    inline size_t size() { return _size; }

    /**
     * @brief push_back function
     *
     * @param key the key to be pushed back
     */
    inline void push_back(T key) {
        std::shared_ptr<node> nn = std::make_shared<node>(key);
        if (!root) {
            root = nn;
            tail = nn;
            _size++;
            return;
        } else {
            tail->next = nn;
            nn->prev = tail;
            tail = nn;
            _size++;
        }
    }

    /**
     * @brief push_front function
     *
     * @param key the key to be pushed front
     */
    inline void push_front(T key) {
        std::shared_ptr<node> nn = std::make_shared<node>(key);
        if (!root) {
            root = nn;
            tail = nn;
            _size++;
            return;
        } else {
            root->prev = nn;
            nn->next = root;
            root = nn;
            _size++;
        }
    }

    /**
     * @brief top function
     *
     * @return T the top of the dequeue
     */
    inline T front() { return root->val; }

    /**
     * @brief back function
     *
     * @return T the back of the dequeue
     */
    inline T back() { return tail->val; }

    /**
     * @brief pop_front function
     * removes the front from the dequeue
     */
    inline void pop_front() {
        root = root->next;
        root->prev = nullptr;
        _size--;
    }

    /**
     * @brief pop_back function
     *removes the back from the queue
     */
    inline void pop_back() {
        tail = tail->prev;
        tail->next = nullptr;
        _size--;
    }

    class Iterator;

    /**
     * @brief pointer to the front of the dequeue
     *
     * @return Iterator
     */
    inline Iterator begin() { return Iterator(root); }

    /**
     * @brief pointer to the end of the dequeue
     *
     * @return Iterator
     */
    inline Iterator end() { return Iterator(nullptr); }
};

/**
 * @brief Iterator class
 */
template <typename T> class dequeue_list<T>::Iterator {
  private:
    std::shared_ptr<node> curr_root;

  public:
    /**
     * @brief Construct a new Iterator object
     *
     * @param q dequeue_list pointer
     */
    explicit Iterator(const std::shared_ptr<node>& q) noexcept : curr_root(q) {}

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
