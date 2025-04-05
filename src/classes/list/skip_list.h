#ifndef SKIP_LIST_H
#define SKIP_LIST_H

#ifdef LINKED_LIST_VISUALIZATION_H
#include "../../visualization/list_visual/linked_list_visualization.h"
#endif

#ifdef __cplusplus
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#endif

/**
 *@brief skip_list class.
 */

template <typename T> class skip_list {
  public:
    /**
     *@brief skip_list constructor.
     *@param __MAX_LEVEL: max height of the list.
     *@param __PROB: probability of increasing the height each time(by default it
     *should be 0.5).
     */
    inline explicit skip_list(int MAX_LEVEL, float PROB) {
        try {
            if (MAX_LEVEL < 5) {
                MAX_LEVEL = MAX_LEVEL;
                root = std::make_shared<node>(0, MAX_LEVEL);
            } else {
                throw std::invalid_argument("Max level value is too high");
            }
            if (PROB > 0.0 && PROB < 1.0) {
                PROB = PROB;
            } else if (PROB >= 1.0) {
                throw std::invalid_argument("Probability value is greater or equal to 1");
            } else if (PROB <= 0.0) {
                throw std::invalid_argument("Probability value is smaller or equal to 0");
            }
        } catch (std::invalid_argument& e) {
            std::cerr << e.what() << '\n';
            return;
        }
    }

    /**
     * @brief Copy constructor for the skip_list class
     *
     * @param s
     */
    inline skip_list(const skip_list& s)
        : level(s.level), PROB(s.PROB), MAX_LEVEL(s.MAX_LEVEL), root(s.root) {}

    /**
     * @brief operator = for the skip_list class
     * @param s the list we want to copy
     * @return skip_list&
     */
    inline skip_list& operator=(const skip_list& s) {
        level = s.level;
        PROB = s.PROB;
        MAX_LEVEL = s.MAX_LEVEL;
        root = s.root;
        return *this;
    }

    /**
     * @brief Destroy the skip list object
     */
    inline ~skip_list() noexcept {}

    /**
     *@brief insert function.
     *@param key: key to be inserted.
     */
    inline void insert(T key) {
        std::shared_ptr<node> head = root;
        std::vector<std::shared_ptr<node>> update(MAX_LEVEL + 1, nullptr);

        for (int i = level; i >= 0; i--) {
            while (head->next[i] != nullptr && head->next[i]->key < key) {
                head = head->next[i];
            }
            update[i] = head;
        }

        head = head->next[0];
        if (!head || head->key != key) {
            int lvl = rand_lvl();
            if (lvl > level) {
                for (int i = level + 1; i < lvl + 1; i++) {
                    update[i] = root;
                }
                level = lvl;
            }

            std::shared_ptr<node> nn = std::make_shared<node>(key, lvl);
            for (int i = 0; i <= lvl; i++) {
                nn->next[i] = update[i]->next[i];
                update[i]->next[i] = nn;
            }
        }
    }

    class Iterator;

    /**
     * @brief pointer that points to the first element of the list
     *
     * @return Iterator
     */
    inline Iterator begin() { return Iterator(root->next[0]); }

    /**
     * @brief pointer that points to the last element of the list
     *
     * @return Iterator
     */
    inline Iterator end() { return Iterator(nullptr); }

    /**
     *@brief remove function.
     *@param key: key to be removed(if exist).
     */
    inline void remove(T key) {
        std::shared_ptr<node> x = root;
        std::vector<std::shared_ptr<node>> update(MAX_LEVEL + 1, nullptr);

        for (int64_t i = level; i >= 0; i--) {
            while (x->next[i] && x->next[i]->key < key) {
                x = x->next[i];
            }
            update[i] = x;
        }

        x = x->next[0];
        if (x && x->key == key) {
            for (int64_t i = 0; i <= level; i++) {
                if (update[i]->next[i] != x) {
                    break;
                }
                update[i]->next[i] = x->next[i];
            }
            while (level > 0 && root->next[level] == nullptr) {
                level--;
            }
        }
    }

    /**
     *@brief search function.
     *@param key: key to be searched.
     *@returns true if the key exists in the list.
     */
    inline bool search(T key) {
        std::shared_ptr<node> x = root;
        for (int64_t i = level; i >= 0; i--) {
            while (x->next[i] && x->next[i]->key < key) {
                x = x->next[i];
            }
        }
        x = x->next[0];
        if (x && x->key == key) {
            return true;
        }
        return false;
    }

    /**
     * @brief visualize function
     * returns a .dot file that can be previewd with graphviz plugin in vscode
     */

#ifdef ENABLE_LIST_VISUALIZATION
    inline void visualize() {
        std::string generated = this->generate();
        linked_list_visualization::visualize(generated);
    }
#endif

    /**
     *@brief operator << for skip_list<T> class.
     */
    inline friend std::ostream& operator<<(std::ostream& out, skip_list<T>& l) {
        std::shared_ptr<node> root = l.root;
        out << "{";
        for (int i = 0; i <= l.level; i++) {
            out << i << ": ";
            std::shared_ptr<node> curr = l.root->next[i];
            while (curr != nullptr) {
                out << curr->key << " ";
                curr = curr->next[i];
            }
            out << '\n';
        }
        out << "}" << '\n';
        return out;
    }

  private:
    int MAX_LEVEL{};
    float PROB{};

    /**
     * @brief struct for the node
     * @param key: the value of the node
     * @param next: vector of pointers to the next
     */
    struct node {
        T key;
        std::vector<std::shared_ptr<node>> next;
        node(T key, int level, void* value = nullptr) : key(key) {
            for (int i = 0; i < level + 1; i++) {
                next.push_back(nullptr);
            }
        }
    };

    int rand_lvl() {
        float r = (float)rand() / RAND_MAX;
        int lvl = 0;
        while (r < PROB && lvl < MAX_LEVEL) {
            lvl++;
            r = (float)rand() / RAND_MAX;
        }
        return lvl;
    }

    int level{0};
    std::shared_ptr<node> root;

    std::string generate_node(std::string node_val, int levs) {
        std::string gen;
        gen += node_val;
        gen += " [label=\"<";
        gen += std::to_string(levs);
        gen += "> ";
        gen += node_val;
        for (int i = levs - 1; i >= 0; i--) {
            gen += " | <";
            gen += std::to_string(i);
            gen += "> ";
            gen += node_val;
        }
        gen += "\"];";
        gen += '\n';
        return gen;
    }

    std::string generate_edge(std::string prev_val, std::string curr_val, int lev) {
        std::string gen;
        gen += prev_val;
        gen += ':';
        gen += std::to_string(lev);
        gen += " -> ";
        gen += curr_val;
        gen += ':';
        gen += std::to_string(lev);
        gen += " ;\n";
        return gen;
    }

    std::string generate() {
        std::string gen;
        gen += "rankdir=LR;";
        gen += '\n';
        gen += "node [shape=record;]";
        gen += '\n';
        std::unordered_set<std::string> S;
        int m_level = std::min(level, (int)root->next.size()); // See if this is a parameter
        gen += generate_node("root", m_level + 1);
        gen += generate_node("NULL", m_level + 1);
        gen += generate_edge("root", "NULL", m_level + 1);
        for (int i = m_level; i >= 0; i--) {
            std::shared_ptr<node> head = root;
            head = head->next[i];
            std::string prev_val = "root";
            std::string head_key;
            while (head) {
                if (std::is_same_v<T, std::string>) {
                    head_key = head->key;
                } else {
                    head_key = to_string(head->key);
                }
                if (S.find(head_key) == S.end()) {
                    S.insert(head_key);
                    gen += generate_node(head_key, i);
                }
                gen += generate_edge(prev_val, head_key, i);
                prev_val = head_key;
                head = head->next[i];
            }
            gen += generate_edge(prev_val, "NULL", i);
        }
        return gen;
    }
};

/**
 * @brief Iterator class
 */
template <typename T> class skip_list<T>::Iterator {
  private:
    std::shared_ptr<node> ptr;

  public:
    /**
     * @brief Construct a new Iterator object
     * @param ptr: pointer to the node
     */
    explicit Iterator(std::shared_ptr<node> ptr) noexcept : ptr(ptr) {}

    /**
     * @brief = operator for Iterator type*
     *
     * @param current  pointer to the node
     * @return Iterator&
     */
    Iterator& operator=(std::shared_ptr<node> current) {
        this->ptr = current;
        return *(this);
    }

    /**
     * @brief operator ++
     *
     * @return Iterator
     */
    Iterator& operator++() {
        if (ptr != nullptr) {
            ptr = ptr->next[0];
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
     * @param it pointer to the node
     * @return bool
     */
    bool operator!=(const Iterator& it) { return ptr != it.ptr; }

    /**
     * @brief operator * for type Iterator
     *
     * @return T
     */
    T operator*() { return ptr->key; }
};

#endif
