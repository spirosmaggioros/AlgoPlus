/**
 * Please have in mind that this header is created for C++20 only
 * This header file is not included at the core code of AlgoPlus
 * as it was created for experimental purposes
 */

#ifndef TABLE_H
#define TABLE_H

#ifdef __cplusplus
#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <ranges>
#endif

template <typename T> class table {
  private:
    struct node {
        T info;
        size_t _idx;
        std::shared_ptr<node> ptr_right;
        std::shared_ptr<node> ptr_left;
        explicit node() : ptr_right(nullptr), ptr_left(nullptr), _idx(0) {}
    };

    size_t _size;
    std::shared_ptr<node> root;
    std::shared_ptr<node> tail;
    std::shared_ptr<node> _itr;

  public:
    explicit table() noexcept : root(nullptr), tail(nullptr), _itr(nullptr), _size(0) {}
    table(const table<T>& t) noexcept : root(t.root), tail(t.tail), _size(t._size) {}

    size_t size() { return this->_size; }

    template <typename... Args> void push_front(Args&&... keys);

    template <typename... Args> void push_back(Args&&... keys);

    void pop_front();

    void pop_back();

    bool empty();

    std::vector<T> vectorize();

    class Iterator;

    Iterator begin() { return Iterator(root); }

    Iterator end() { return Iterator(nullptr); }

    T& operator[](const size_t& index) {
        assert(index < _size && index >= 0);
        if (index == 0) {
            return this->root->info;
        }

        if (index == _size - 1) {
            return this->tail->info;
        }

        if (_itr->_idx < index) {
            while (_itr->_idx != index) {
                _itr = _itr->ptr_right;
            }
        } else {
            while (_itr->_idx != index) {
                _itr = _itr->ptr_left;
            }
        }

        return _itr->info;
    }

    table<T>& operator=(const table<T>& t) {
        if (this == &t) [[unlikely]] {
            return *this;
        } else [[likely]] {
            this->root = t.root;
            this->tail = t.tail;
        }
        return *(this);
    }
    friend std::ostream& operator<<(std::ostream& out, const table<T>& t) {
        if (t.root == nullptr) [[unlikely]] {
            return out;
        } else [[likely]] {
            std::shared_ptr<node> tmp = t.root;
            while (tmp != nullptr) {
                out << tmp->info << " ";
                tmp = tmp->ptr_right;
            }
        }
        return out;
    }

  protected:
    void _check_update() {
        if (this->root == nullptr) {
            return;
        }
        std::shared_ptr<node> tmp = this->root;
        size_t idx = 0;
        while (tmp != nullptr) {
            tmp->_idx = idx++;
            tmp = tmp->ptr_right;
        }
    }
};

template <typename T> template <typename... Args> void table<T>::push_front(Args&&... keys) {
    auto _insert = [&](const T&& key) -> void {
        std::shared_ptr<node> nn = std::make_shared<node>();
        nn->info = key;
        if (this->root == nullptr) [[unlikely]] {
            nn->_idx = 0;
            this->root = nn;
            this->tail = nn;
            this->_itr = nn;
            this->_size++;
            return;
        } else [[likely]] {
            std::shared_ptr<node> tmp = root;
            nn->_idx = 0;
            this->root = nn;
            this->root->ptr_right = tmp;
            tmp->ptr_left = this->root;
        }
        this->_size++;
    };
    (std::invoke(_insert, std::forward<Args>(keys)), ...);
    _check_update();
}

template <typename T> template <typename... Args> void table<T>::push_back(Args&&... keys) {
    auto _insert = [&](const T&& key) -> void {
        std::shared_ptr<node> nn = std::make_shared<node>();
        nn->info = key;
        if (this->root == nullptr) [[unlikely]] {
            nn->_idx = 0;
            this->root = nn;
            this->tail = nn;
            this->_itr = nn;
            this->_size++;
            return;
        } else [[likely]] {
            std::shared_ptr<node> tmp = tail;
            nn->_idx = tail->_idx + 1;
            this->tail = nn;
            tmp->ptr_right = tail;
            this->tail->ptr_left = tmp;
        }
        this->_size++;
    };
    (std::invoke(_insert, std::forward<Args>(keys)), ...);
}

template <typename T> void table<T>::pop_back() {
    if (this->tail == nullptr) [[unlikely]] {
        return;
    } else [[likely]] {
        this->tail = this->tail->ptr_left;
    }
    _size--;
}

template <typename T> void table<T>::pop_front() {
    if (this->root == nullptr) [[unlikely]] {
        return;
    } else [[likely]] {
        this->root = this->root->ptr_right;
    }
    _size--;
    _check_update();
}

template <typename T> bool table<T>::empty() {
    return this->_size == 0;
}

template <typename T> std::vector<T> table<T>::vectorize() {
    if (this->root == nullptr) {
        return {};
    }
    std::vector<T> vec;
    for (auto it = this->begin(); it != this->end(); it++) {
        vec.push_back(*(it));
    }
    return vec;
}

template <typename T> class table<T>::Iterator {
  private:
    std::shared_ptr<node> curr_root;

  public:
    explicit Iterator(const std::shared_ptr<node>& ptr) noexcept : curr_root(ptr) {}

    Iterator& operator=(std::shared_ptr<node> current) {
        this->curr_root = current;
        return *(this);
    }

    Iterator& operator++() {
        if (curr_root) {
            curr_root = curr_root->ptr_right;
        }
        return *(this);
    }

    Iterator operator++(int) {
        Iterator it = *this;
        ++*(this);
        return it;
    }

    Iterator& operator--() {
        if (curr_root) {
            curr_root = curr_root->ptr_left;
        }
        return *(this);
    }

    Iterator operator--(int) {
        Iterator it = *this;
        --*(this);
        return it;
    }

    bool operator!=(const Iterator& it) { return curr_root != it.curr_root; }

    T operator*() { return curr_root->info; }
};

#endif
