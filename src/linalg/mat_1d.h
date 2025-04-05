#ifndef MAT_1D_H
#define MAT_1D_H

#ifdef __cplusplus
#include <cassert>
#include <climits>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <vector>
#include "../algorithms/sorting/quick_sort.h"
#endif

/**
 *@brief Class for 1-dimensional Matrix
 *
 */
template <typename T, size_t SIZE> class Mat1d {
  private:
    T* arr;
    size_t _size;

  public:
    /**
     *@brief constructor for Mat1d class
     *
     */
    explicit Mat1d(std::vector<T> v = {}) : _size(SIZE) {
        arr = new T[_size];
        if (!v.empty()) {
            try {
                if (v.size() != _size) {
                    throw std::logic_error("Initializer array don't have the same size "
                                           "as the constructed array");
                } else {
                    for (size_t i = 0; i < _size; i++) {
                        arr[i] = v[i];
                    }
                }
            } catch (std::logic_error& e) {
                std::cerr << e.what() << '\n';
            }
        }
    }

    /**
     *@brief constructor for Mat1d class with initializer list
     *@param il the initializer list that we want to use to initialize the array
     */
    explicit Mat1d(std::initializer_list<T> il) : _size(SIZE) {
        arr = new T[_size];
        if (il.size() != _size) {
            throw std::logic_error("Initializer list doesn't have the same size as "
                                   "the constructed array");
        }
        std::copy(il.begin(), il.end(), arr);
    }

    /**
     *@brief constructor for Mat1d class with input value
     *@param val the value that we want all the elements of the array to have
     */
    explicit Mat1d(const T val) noexcept : _size(SIZE) {
        arr = new T[_size];
        for (size_t i = 0; i < _size; ++i) {
            arr[i] = val;
        }
    }

    /**
     *@brief copy constructor for Mat1d class
     *@param mat the matrix we want to copy
     *@return Mat1d
     */
    explicit Mat1d(Mat1d& mat) : _size(SIZE) {
        try {
            if (mat.size() != _size) {
                throw std::logic_error("Tried to copy matrices with different sizes");
            }
            this->arr = new T[_size];
            for (size_t i = 0; i < _size; i++) {
                this->arr[i] = mat.arr[i];
            }
        } catch (std::logic_error& e) {
            std::cerr << e.what() << '\n';
        }
    }

    /**
     *@brief copy constructor for Mat1d class
     *@param mat the matrix we want to copy
     *@return Mat1d&
     */
    Mat1d& operator=(Mat1d& mat) {
        if (this == &mat) {
            return *(this);
        }
        assert(mat.size() == this->_size);
        for (size_t i = 0; i < _size; i++) {
            this->arr[i] = mat.arr[i];
        }
        return *(this);
    }

    /**
     *@brief destructor of Mat1d class
     *
     */
    ~Mat1d() { delete[] arr; }

    /**
     *@brief size function
     *@return size_t the size of the array
     *
     */
    size_t size() const { return this->_size; }

    /**
     *@brief Iterator begin for Mat1d class
     *
     */
    typedef T* iterator;
    T* begin() { return &arr[0]; }
    // Iterator begin() { return Iterator(0, _size, arr); }

    /**
     *@brief Iterator end for Mat1d class
     *
     */
    T* end() { return &arr[0] + SIZE; }
    // Iterator end() { return Iterator(_size, _size, arr); }

    /**
     *@brief operator[] for Mat1d class
     *@return T the value of the array to that index
     */
    T& operator[](const size_t index) {
        assert(index < _size);
        return arr[index];
    }

    /**
     *@brief operator == for Mat1d class
     *@param mat the matrix to compare
     *@return true if &this is equal to mat
     *@return false otherwise
     */
    bool operator==(const Mat1d<T, SIZE>& mat) const {
        if (mat.size() != _size) {
            return false;
        }

        for (size_t i = 0; i < _size; i++) {
            if (mat.arr[i] != this->arr[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     *@brief operator != for Mat1d class
     *@param mat the matrix to compare
     *@return true if &this is not equal to mat
     *@return false otherwise
     */
    bool operator!=(const Mat1d<T, SIZE>& mat) const { return !(*this == mat); }

    /**
     *@brief operator << for Mat1d class
     *@param mat the matrix
     *
     */
    friend std::ostream& operator<<(std::ostream& out, const Mat1d& mat) {
        out << '[';
        for (size_t i = 0; i < mat.size(); i++) {
            out << mat.arr[i];
            if (i != mat.size() - 1) {
                out << " ";
            }
        }
        out << ']' << '\n';
        return out;
    }
};

/**
 * @brief Template specialization for comparing Mat1d objects of different sizes
 */
template <typename T, size_t SIZE1, size_t SIZE2>
bool operator==(const Mat1d<T, SIZE1>& mat1, const Mat1d<T, SIZE2>& mat2) {
    return false; // Matrices of different sizes are not equal
}

template <typename T, size_t SIZE1, size_t SIZE2>
bool operator!=(const Mat1d<T, SIZE1>& mat1, const Mat1d<T, SIZE2>& mat2) {
    return !(mat1 == mat2); // Uses the == operator specialization
}

#endif
