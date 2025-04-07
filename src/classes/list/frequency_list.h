#ifndef FREQUENCY_LIST_H
#define FREQUENCY_LIST_H

#ifdef ENABLE_LINKED_LIST_VISUALIZATION
#include "../../visualization/list_visual/linked_list_visualization.h"
#endif

#ifdef __cplusplus
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#endif

/**
 * @class frequency_list
 * @template typename T
 * @brief Self-learning Frequency List that maintains a list of elements in
 * descending order of their frequency.
 *
 * @details
 * The frequency_list is a container that maintains a list of elements and their
 * frequencies. Each element is stored in a node along with the count of the
 * number of times it occurs in the list. The frequency_list is useful when it
 * is important to keep track of the number of times an item appears in the
 * list.
 *
 * This class should be used for, for example, tracking user choices, making
 * them more accessible for the user in the future, improving their experience.
 */
template <typename T> class frequency_list {
  public:
    /**
     * @brief Constructs a frequency_list object with an optional initializer list
     * of elements.
     *
     * The frequency_list object is constructed with an optional initializer list
     * of elements. If the initializer list is not empty, the elements are added
     * to the frequency_list in the same order as they appear in the vector.
     *
     * @tparam T The type of elements stored in the frequency_list.
     * @param data An optional initializer list of elements.
     */
    inline explicit frequency_list(std::vector<T> data = {}) noexcept
        : head(nullptr), tail(nullptr) {
        if (!data.empty()) {
            for (const auto& item : data) {
                this->push_back(item);
            }
        }
    }

    /**
     * @brief Copy constructor for the frequency_list class.
     *
     * This constructor creates a new frequency_list object by copying the
     * elements from another frequency_list object. The elements are copied in the
     * same order as they appear in the original frequency_list object.
     *
     * @param list The frequency_list object to copy from.
     */
    inline frequency_list(const frequency_list<T>& list) : head(nullptr), tail(nullptr), size(0) {
        for (auto iter = list.head; iter != nullptr; iter = iter->next) {
            this->push_back(iter->data);
        }
    }

    class Iterator;

    /**
     * @brief Adds an element to the back of the frequency list.
     *
     * This function adds an element to the back of the frequency list. If the
     * frequency list is empty, it creates a new node with the given data and
     * frequency of 1 and sets it as the head. If the element already exists in
     * the frequency list, it increments its frequency. Otherwise, it creates a
     * new node with the given data and frequency of 1, and appends it to the end
     * of the frequency list.
     *
     * @param data The data to be added to the frequency list.
     */
    void push_back(T data);

    /**
     * @brief Adds an element to the front of the frequency list.
     *
     * If the frequency list is not empty, it updates the previous pointer
     * of the current head node to the new node. Then
     * it sets the new node as the new head of the frequency list.
     *
     * @param data The data to be added to the frequency list.
     */
    void push_front(T data);

    /**
     * @brief Searches for a given key in the frequency list.
     *
     * This function searches for a given key in the frequency list. If the key is
     * found, it increments the frequency of the corresponding node and returns
     * true. If the key is not found, it returns false.
     *
     * @param key The key to search for in the frequency list.
     * @return Returns true if the key is found and the frequency is incremented,
     * false otherwise.
     */
    bool search(T key);

    /**
     * @brief Gets the frequency of a key in the frequency list.
     *
     * @param key The key to get the frequency of.
     * @return The frequency of the key, or -1 if the key is not found.
     */
    int64_t get_frequency(T key);

    /**
     * @brief Removes the first occurrence of a given key in the frequency list.
     *
     * If the key is found, it removes the corresponding node and updates the
     * pointers. If the head node is the one to be removed, the head pointer is
     * updated accordingly. If the node has a previous and/or next node, their
     * pointers are updated to connect them together.
     *
     * @param key The key to be removed from the frequency list.
     */
    void erase(T key);

    /**
     * @brief Resets the frequency of all nodes in the frequency list to 1.
     */
    void reset_frequency();

    /**
     *@brief Returns all the elements of the list
     *@return std::vector<T> the elements of the list
     *
     */
    std::vector<std::pair<T, int64_t>> elements();

    /**
     * @brief Checks if the frequency list is empty.
     *
     * @return Returns true if the frequency list is empty, false otherwise.
     */
    bool empty() { return head == nullptr; }

    /**
     * @brief Get an iterator pointing to the beginning of the frequency list.
     *
     * The Iterator provides a way to traverse the frequency list.
     *
     * @return An Iterator object pointing to the beginning of the frequency list.
     */
    Iterator begin() { return Iterator(head); }

    /**
     * @brief Get an iterator pointing to the end of the frequency list.
     *
     * The end iterator is used to indicate the position after the last element of
     * the frequency list, and can be used to determine when the end of the
     * frequency list has been reached while iterating.
     *
     * @return An Iterator object pointing to the end of the frequency list.
     */
    Iterator end() { return Iterator(nullptr); }

    /**
     *
     *@brief visualize function for frequency_list Class
     *returns a .dot file that can be previewd with graphviz plugin in vscode
     */
    void visualize();

    /**
     * @brief Overloaded output stream insertion operator for the frequency_list
     * class.
     *
     * This operator allows the frequency_list object to be printed to the output
     * stream. It iterates through the frequency_list and outputs each element
     * along with its frequency in the format: element(frequency)
     *
     * @param os The output stream to write to.
     * @param flist The frequency_list object to be printed.
     * @return The output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const frequency_list<T>& flist) {
        auto iterator = flist.head;
        while (iterator != nullptr) {
            os << iterator->data << "(" << iterator->freq << ") ";
            iterator = iterator->next;
        }
        return os;
    }

  private:
    struct node {
        T data;
        int64_t freq;
        std::shared_ptr<node> next;
        std::weak_ptr<node> prev;

        node(T data, int64_t freq, std::weak_ptr<node> prev, std::shared_ptr<node> next)
            : data(data), freq(freq), prev(prev), next(next) {}

        void add_freq() { freq++; }
        void reset_freq() { freq = 1; }
    };

    std::shared_ptr<node> head;
    node* tail;
    uint64_t size{0};

    /**
     * @brief Updates the order of nodes in the frequency list based on their
     * frequencies.
     *
     * This function updates the order of nodes in the frequency list based on
     * their frequencies. It compares each node with its adjacent node and swaps
     * them if the frequency of the current node is less than the frequency of the
     * next node. After swapping, it updates the current node to point to its
     * previous node. If the previous node exists, it becomes the new current
     * node. If the previous node does not exist, the traversal stops.
     */
    void update_list_sequence();

    /**
     * @brief Swaps the positions of two nodes in the frequency list.
     *
     * If either of the nodes is null or they are not adjacent to each other, the
     * function does nothing.
     *
     * The function modifies the pointers of the adjacent nodes and updates the
     * head if necessary.
     *
     * @param a The first node to be swapped.
     * @param b The second node to be swapped.
     */
    void swap_nodes(std::shared_ptr<node> a, std::shared_ptr<node> b);

    /**
     *
     *@brief generate a string that contains all the information for the
     *visualization
     *@return std::string the information for the .dot file
     */
    std::string generate();
};

template <typename T>
inline void frequency_list<T>::swap_nodes(std::shared_ptr<node> a, std::shared_ptr<node> b) {
    if (a == nullptr || b == nullptr || a->next != b)
        return;

    // Attach b's next to a and a's prev to b
    a->next = b->next;
    b->prev = a->prev;

    // If b's next exists, attach a to its prev
    if (b->next)
        b->next->prev = a;

    // If a's prev exists(i.e., a is not the head), attach b to its next
    if (a->prev.lock())
        a->prev.lock()->next = b;

    // Now, attach a to b's next and b to a's prev
    b->next = a;
    a->prev = b;

    // If the node a was the head node, update the head to b
    if (head == a)
        head = b;
}

template <typename T> inline void frequency_list<T>::update_list_sequence() {
    auto current_node = head;

    while (current_node != nullptr && current_node->next != nullptr) {
        if (current_node->freq < current_node->next->freq) {
            swap_nodes(current_node, current_node->next);

            if (current_node->prev.lock())
                current_node = current_node->prev.lock();

        } else
            current_node = current_node->next;
    }
}

template <typename T> inline void frequency_list<T>::reset_frequency() {
    auto iterator = head;

    while (iterator != nullptr) {
        iterator->reset_freq();
        iterator = iterator->next;
    }
}

template <typename T> inline void frequency_list<T>::push_front(T data) {
    auto Iterator = head;

    // Iterating over the list to find if the same data is already present
    while (Iterator != nullptr) {
        if (Iterator->data == data) {
            // if data is found, increase frequency and re-sort the list
            Iterator->freq++;
            update_list_sequence();
            return;
        }
        Iterator = Iterator->next;
    }

    // if data is not found in the list, add new node at front
    auto new_node = std::make_shared<node>(data, 1, std::weak_ptr<node>(), head);

    if (head != nullptr) {
        head->prev = new_node;
    }

    head = new_node;

    size++;
}

template <typename T> inline void frequency_list<T>::erase(T key) {
    auto iterator = head;

    // special case: deleting the head node
    if (head != nullptr && head->data == key) {
        head = head->next;
        if (head)
            head->prev.reset();
        size--;
        return;
    }

    // iterate over the rest nodes
    while (iterator != nullptr) {
        if (iterator->data == key) {
            // connect previous node and next node
            if (auto sharedPrev = iterator->prev.lock()) {
                sharedPrev->next = iterator->next;
            }
            if (iterator->next) {
                iterator->next->prev = iterator->prev;
            }
            // deleting the node by falling out of scope
            size--;
            return;
        }
        iterator = iterator->next;
    }
}

template <typename T> inline int64_t frequency_list<T>::get_frequency(T key) {
    auto iterator = head;

    while (iterator != nullptr) {
        if (iterator->data == key) {
            return iterator->freq;
        }

        iterator = iterator->next;
    }

    return -1;
}

template <typename T> inline bool frequency_list<T>::search(T key) {
    auto iterator = head;

    while (iterator != nullptr) {
        if (iterator->data == key) {
            iterator->add_freq();
            update_list_sequence();
            return true;
        }

        iterator = iterator->next;
    }

    return false;
}

template <typename T> inline void frequency_list<T>::push_back(T data) {
    if (this->empty()) {
        head = std::make_shared<node>(data, 1, std::weak_ptr<node>(), nullptr);
        return;
    }

    auto iterator = head;
    while (iterator) {
        if (iterator->data == data) {
            iterator->add_freq();
            update_list_sequence();
            return;
        }
        if (!iterator->next)
            break;
        iterator = iterator->next;
    }

    auto newNode = std::shared_ptr<node>(new node(data, 1, iterator, nullptr));
    if (iterator)
        iterator->next = newNode;
    size++;
}

template <typename T> inline std::vector<std::pair<T, int64_t>> frequency_list<T>::elements() {
    std::vector<std::pair<T, int64_t>> ans;
    std::shared_ptr<node> root = head;
    while (root) {
        ans.push_back(std::make_pair(root->data, root->freq));
        root = root->next;
    }
    return ans;
}

template <typename T> inline std::string frequency_list<T>::generate() {
    std::string gen = "";
    gen += "rankdir=LR;";
    gen += '\n';
    gen += "node [shape=record;]";
    gen += '\n';
    std::vector<std::pair<T, int64_t>> els = this->elements();
    if (std::is_same_v<T, std::string> || std::is_same_v<T, char>) {
        for (auto& x : els) {
            gen += x.first;
            gen += " [label=<{ ";
            gen += x.first;
            gen += " | ";
            gen += x.second;
            gen += " | }>] ;";
            gen += '\n';
        }

        std::shared_ptr<node> curr = head;
        while (curr->next) {
            gen += curr->data;
            gen += ":ref -> ";
            gen += curr->next->data;
            gen += ":data [arrowhead=vee, arrowtail=dot, dir=both];";
            gen += '\n';

            gen += curr->next->data;
            gen += ":data -> ";
            gen += curr->data;
            gen += ":ref [arrowhead=vee, arrowtail=dot, dir=both];";
            gen += '\n';

            curr = curr->next;
        }
    } else {
        for (auto& x : els) {
            std::cout << std::to_string(x.first) << " " << std::to_string(x.second) << '\n';
            gen += std::to_string(x.first);
            gen += " [label=<{ ";
            gen += std::to_string(x.first);
            gen += " | ";
            gen += std::to_string(x.second);
            gen += " | }>] ;";
            gen += '\n';
        }

        std::shared_ptr<node> curr = head;
        while (curr->next) {
            gen += std::to_string(curr->data);
            gen += ":ref -> ";
            gen += std::to_string(curr->next->data);
            gen += ":data [arrowhead=vee, arrowtail=dot, dir=both];";
            gen += '\n';

            gen += std::to_string(curr->next->data);
            gen += ":data -> ";
            gen += std::to_string(curr->data);
            gen += ":ref [arrowhead=vee, arrowtail=dot, dir=both];";
            gen += '\n';

            curr = curr->next;
        }
    }
    return gen;
}

#ifdef ENABLE_LINKED_LIST_VISUALIZATION
template <typename T> inline void frequency_list<T>::visualize() {
    std::string generated = this->generate();
    linked_list_visualization::visualize(generated);
}
#endif

template <typename T> class frequency_list<T>::Iterator {
  private:
    std::shared_ptr<node> current;

  public:
    /**
     * @brief Iterator class for a linked list.
     *
     * The Iterator class provides an iterator for traversing a linked list.
     */
    explicit Iterator(const std::shared_ptr<node>& ptr) noexcept : current(ptr) {}

    /**
     * @brief Assignment operator for the Iterator class.
     *
     * It assigns the value of the given shared pointer to the current node of the
     * iterator and returns a reference to the updated iterator.
     *
     * @param ptr The shared pointer to assign.
     * @return A reference to the updated iterator.
     */
    Iterator& operator=(const std::shared_ptr<node>& ptr) {
        current = ptr;
        return *this;
    }

    /**
     * @brief Pre-increment operator for the Iterator class.
     *
     * This operator overloads the pre-increment operator (++).
     * It increments the iterator to the next node and returns a reference to the
     * updated iterator.
     *
     * @return A reference to the updated iterator.
     */
    Iterator& operator++() {
        if (current) {
            current = current->next;
        }
        return *this;
    }

    /**
     * @brief Post-increment operator for the Iterator class.
     *
     * This operator overloads the post-increment operator (++).
     * It increments the iterator to the next node and returns a copy of the
     * iterator before the increment.
     *
     * @return A copy of the iterator before the increment.
     */
    Iterator operator++(int) {
        Iterator tmp = *(this);
        ++*(this);
        return tmp;
    }

    /**
     *@brief Pre-increment -- operator for the Iterator class.
     *This operator overloads the pre-increment operator (--).
     * It increments the iterator to the next node and returns a reference to the
     *updated iterator.
     *
     * @return Iterator&
     *
     */
    Iterator& operator--() {
        if (current) {
            current = current->prev;
        }
        return *(this);
    }

    /**
     *@brief Post-increment --operator for the Iterator class.
     *This operator overloads the post-increment operator (--).
     * It increments the iterator to the next node and returns a copy of the
     *iterator before the increment.
     *
     *@return Iterator
     */
    Iterator operator--(int) {
        Iterator tmp = *(this);
        --*(this);
        return tmp;
    }

    /**
     * @brief Equality operator for the Iterator class.
     *
     * This operator checks if two iterators are equal by comparing their current
     * nodes.
     *
     * @param it The other Iterator object to compare with.
     * @return true if the iterators are equal, false otherwise.
     */
    bool operator==(const Iterator& it) { return it.current == this->current; }

    /**
     * @brief Inequality operator for the Iterator class.
     *
     * This operator checks if two iterators are not equal by comparing their
     * current nodes.
     *
     * @param it The other Iterator object to compare with.
     * @return true if the iterators are not equal, false otherwise.
     */
    bool operator!=(const Iterator& it) { return it.current != this->current; }

    /**
     * @brief Dereference operator for the Iterator class.
     *
     * @return The data pointed to by the current node.
     */
    T operator*() { return current->data; }
};

#endif // FREQUENCY_LIST_H
