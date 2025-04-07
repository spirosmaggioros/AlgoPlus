#ifndef TRIE_H
#define TRIE_H

#ifdef __cplusplus
#include <iostream>
#include <memory>
#include <vector>
#endif

/**
 *@brief trie class
 */
class trie {
  private:
    struct node {
        std::shared_ptr<node> characters[26]{nullptr};
        bool end_word{false};
    };

    std::shared_ptr<node> root;
    size_t _size{};

  public:
    /**
     * @brief Construct a new trie object
     *
     * @param v : vector of strings for initializer.
     */
    inline explicit trie(std::vector<std::string> v = {}) noexcept
        : root(std::make_shared<node>()) {
        if (!v.empty()) {
            for (auto& x : v) {
                this->insert(x);
            }
        }
    }

    /**
     * @brief Copy constructor for trie class
     * @param t the tree we want to copy
     */
    inline explicit trie(const trie& t) : root(t.root), _size(t._size) {}

    /**
     * @brief operator = for trie class
     * @param t the tree we want to copy
     * @return trie&
     */
    inline trie& operator=(const trie& t) {
        root = t.root;
        _size = t._size;
        return *this;
    }

    /**
     *@brief empty function.
     *@returns true if the trie is empty.
     */
    inline bool empty() { return root == nullptr; }
    /**
     *@brief insert function.
     *@param key: the key to be inserted.
     */
    inline void insert(std::string key);

    /**
     * @brief size function
     *
     * @return size_t the size of the tree
     */
    inline size_t size() { return _size; }
    /**
     *@brief remove function.
     *@param key: the key to be removed.
     */
    inline void remove(std::string key);
    /**
     *@brief search function.
     *@param key: the key to be searched.
     *@returns true if the word exist in the tree.
     */
    inline bool search(std::string key);

    inline friend std::ostream& operator<<(std::ostream& out, trie& t);

  private:
    /**
     *@brief __children function.
     *checks if a node has children or not.
     */
    bool _children(std::shared_ptr<node> root) {
        for (int64_t i = 0; i < 26; i++) {
            if (root->characters[i]) {
                return true;
            }
        }
        return false;
    }
    /**
     *@brief __remove function.
     *helper function of remove().
     */
    std::shared_ptr<node> _remove(std::shared_ptr<node> current, std::string word, int64_t index) {
        if (word.size() == index) {
            if (current->end_word) {
                current->end_word = false;
                _size--;
            }
            if (_children(current)) {
                return current;
            }
            return nullptr;
        }

        int64_t i = word[index] - 'a';
        if (!current->characters[i]) {
            return nullptr;
        }
        current->characters[i] = _remove(current->characters[i], word, index + 1);
        if (current->characters[i] || _children(current)) {
            return current;
        }
        return nullptr;
    }
};

void trie::insert(std::string key) {
    std::shared_ptr<node> current = root;
    for (auto& c : key) {
        int64_t index = c - 'a';
        if (!current->characters[index]) {
            current->characters[index] = std::make_shared<node>();
        }
        current = current->characters[index];
    }
    current->end_word = true;
    _size++;
}

void trie::remove(std::string key) {
    root = _remove(root, key, 0);
}

bool trie::search(std::string key) {
    std::shared_ptr<node> current = root;
    for (auto& c : key) {
        int64_t index = c - 'a';
        if (!current->characters[index]) {
            return false;
        }
        current = current->characters[index];
    }
    return current->end_word;
}

#endif
