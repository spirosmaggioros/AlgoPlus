#ifndef MIN_HEAP_H
#define MIN_HEAP_H
#ifdef __cplusplus
#include <climits>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>
#endif

// This is needed to avoid conflict with windows __min macro
#undef __min

/**
 * @brief min heap class
 *
 */
template <typename T> class min_heap {
  private:
    T* arr;
    size_t max_size{};
    size_t heap_size{0};

  public:
    /**
     * @brief Construct a new min heap object
     *
     * @param max_size : maximum size of the heap
     */
    inline explicit min_heap(size_t max_size) noexcept : arr(new T[max_size]) {

        this->max_size = max_size;
    }

    inline ~min_heap() noexcept { delete[] arr; }

    /**
     * @brief parent function
     *
     * @param i the element we want to find the parent
     */
    inline T parent(T i) { return (i - 1) / 2; }

    /**
     * @brief __left function
     *
     * @param i: the element we want to find the left
     */
    inline T _left(T i) { return (2 * i + 1); }

    /**
     * @brief __right function
     *
     * @param i: the element we want to find the right
     */
    inline T _right(T i) { return (2 * i + 2); }

    /**
     * @brief __min function
     * Returns the minimum with heapify
     */
    inline T _min() {
        if (heap_size <= 0) {
            return INT_MAX;
        }
        if (heap_size == 1) {
            heap_size--;
            return arr[0];
        }
        T root = arr[0];
        arr[0] = arr[heap_size - 1];
        heap_size--;
        heapify(0);
        return root;
    }

    /**
     * @brief min function.
     * Returns the minimum of the heap(the first element)
     */
    inline T min() { return arr[0]; }

    inline void decrease_key(T i, T key) {
        arr[i] = key;
        while (i != 0 && arr[parent(i)] > arr[i]) {
            std::swap(arr[i], arr[parent(i)]);
            i = parent(i);
        }
    }

    /**
     * @brief insert function
     *
     * @param key the key to be inserted
     */
    inline void insert(T key) {
        try {
            if (heap_size == max_size) {
                throw std::invalid_argument("Heap reached maximum size");
            } else {
                heap_size++;
                size_t i = heap_size - 1;
                arr[i] = key;
                while (i != 0 && arr[parent(i)] > arr[i]) {
                    std::swap(arr[i], arr[parent(i)]);
                    i = parent(i);
                }
            }
        } catch (std::invalid_argument& e) {
            std::cerr << e.what() << '\n';
            return;
        }
    }

    /**
     * @brief remove function
     *
     * @param key the key to be removed
     */
    inline void remove(T key) {
        decrease_key(key, INT_MIN);
        _min();
    }

    /**
     * @brief heapify function
     *
     * @param i the element we want to heapify from
     */
    inline void heapify(T i) {
        T left = _left(i);
        T right = _right(i);
        T minim = i;
        if (left < heap_size && arr[left] < arr[i]) {
            minim = left;
        }
        if (right < heap_size && arr[right] < arr[i]) {
            minim = right;
        }
        if (minim != i) {
            std::swap(arr[i], arr[minim]);
            heapify(minim);
        }
    }
};

#endif
