#ifndef MAT_2D_H
#define MAT_2D_H

#ifdef __cplusplus
#include <climits>
#include <iostream>
#include <vector>
#include "mat_1d.h"
#endif

/**
 *@brief Class for 2-dimensional Matrix
 *
 */

template <typename T, size_t ROWS, size_t COLS> class Mat2d {
  private:
    T* arr;
    size_t _size;
    size_t _cols;
    size_t _rows;
    Mat1d<T, COLS> mat;

  public:
    /**
     *@brief constructor for Mat2d class
     *
     */
    explicit Mat2d(std::vector<std::vector<T>> v = {})
        : arr(new T[ROWS * COLS]), _cols(COLS), _rows(ROWS), _size(ROWS * COLS) {

        if (!v.empty()) {
            try {
                if (v.size() * v[0].size() != _size || v[0].size() != _cols || v.size() != _rows) {
                    throw std::logic_error("Initializer array don't have the same "
                                           "dimensions as the constructed array");
                }
                for (size_t i = 0; i < v.size(); i++) {
                    for (size_t j = 0; j < v[0].size(); j++) {
                        arr[i * v[0].size() + j] = v[i][j];
                    }
                }
            } catch (std::logic_error& e) {
                std::cerr << e.what() << '\n';
            }
        }
    }

    ~Mat2d() { delete[] arr; }
    /**
     *@brief constructor for Mat2d class with initializer value
     *@param val the value that we want all the index of the array to have
     */
    explicit Mat2d(const T val) noexcept : _size(ROWS * COLS), _cols(COLS), _rows(ROWS) {
        arr = new T[_size];
        for (size_t i = 0; i < _size; i++) {
            arr[i] = val;
        }
    }

    /**
     *@brief copy constructor for Mat2d class
     *@param mat the 2d matrix we want to copy
     *
     */
    explicit Mat2d(const Mat2d& mat) : _size(ROWS * COLS), _cols(COLS), _rows(ROWS) {
        try {
            if (mat.size() != _size || mat.cols() != _cols || mat.rows() != _rows) {
                throw std::logic_error("Tried to copy matrixes with different dimensions");
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
     *@brief operator = for Mat2d class
     *@param mat the matrix we want to copy
     *@return Mat2d&
     */
    Mat2d& operator=(Mat2d& mat) {
        if (this == &mat) {
            return *(this);
        }
        try {
            if (mat.size() != _size || mat.cols() != _cols || mat.rows() != _rows) {
                throw std::logic_error("Tried to copy matrixes with different dimensions");
            }
            for (size_t i = 0; i < _size; i++) {
                this->arr[i] = mat.arr[i];
            }
            return *(this);
        } catch (std::logic_error& e) {
            std::cerr << e.what() << '\n';
        }
        return *(this);
    }

    /**
     *@brief operator (i, j) for Mat2d class
     *@param i index that's pointing to the row
     *@param j index that's pointing to the column
     *@return T the element of arr[i][j]
     */
    T& operator()(size_t i, size_t j) const { return arr[i * _cols + j]; }

    /**
     *@brief operator () for Mat2d class
     *@param i index that's pointing to the row
     *@return std::vector<T> the array of index i
     */
    Mat1d<T, COLS>& operator()(size_t i) {
        size_t index = 0;
        for (size_t j = i * _cols; j < i * _cols + _rows; j++) {
            mat[index++] = arr[j];
        }
        return mat;
    }

    /**
     *@brief size function
     *@return size_t the size of the matrix
     */
    size_t size() const { return _size; }

    /**
     *@brief cols function
     *@return size_t the number of elements in each column
     */
    size_t cols() const { return _cols; }

    /**
     *@brief rows function
     *@return size_t the number of elements in each row
     */
    size_t rows() const { return _rows; }

    class Iterator;
    typedef T* iterator;
    /**
     *@brief begin() Iterator for Mat2d class
     *@return Iterator
     */
    Iterator begin() { return Iterator(arr, _rows, _cols, 0); }

    /**
     *@brief end() Iterator for Mat2d class
     *@return Iterator
     */
    Iterator end() { return Iterator(arr, _rows, _cols, _size); }

    /**
     *@brief operator << for Mat2d class
     */
    friend std::ostream& operator<<(std::ostream& out, Mat2d& mat) {
        out << '[';
        for (size_t i = 0; i < mat.rows(); i++) {
            out << '[';
            for (size_t j = 0; j < mat.cols(); j++) {
                out << mat.arr[i * mat.cols() + j];
                if (j != mat.cols() - 1) {
                    out << " ";
                }
            }
            out << ']';
            if (i != mat.rows() - 1) {
                out << '\n';
            }
        }
        out << ']';
        return out;
    }
};

/**
 *@brief Iterator for Mat2d class
 */
template <typename T, size_t ROWS, size_t COLS> class Mat2d<T, ROWS, COLS>::Iterator {
  private:
    T* arr;
    size_t _rows;
    size_t _cols;
    size_t _index;
    size_t _size;

  public:
    /**
     *@brief constructor for Iterator class
     *@param __arr the input array
     *@param rows the number of rows of the matrix
     *@param cols the number of columns of the matrix
     *@param index the current index, if it's begin() then 0, if it's end() then
     *__size
     */
    explicit Iterator(T* _arr, size_t rows, size_t cols, size_t index) noexcept
        : _rows(rows), _cols(cols), _index(index), _size(rows * cols) {

        arr = new T[_size];
        for (size_t i = 0; i < _size; i++) {
            arr[i] = _arr[i];
        }
    }

    /**
     *@brief operator = for Iterator class
     *@param curr the current array
     *@return Iterator&
     */
    Iterator& operator=(T* curr) {
        for (size_t i = 0; i < _size; i++) {
            arr[i] = curr[i];
        }
        return *(this);
    }

    /**
     *@brief operator ++ for Iterator class
     *@return Iterator&
     */
    Iterator& operator++() {
        if (this->_index < _size) {
            this->_index++;
        }
        return *(this);
    }

    /**
     *@param operator ++ for Iterator class
     *@return Iterator&
     */
    Iterator& operator++(int) {
        Iterator it = *(this);
        ++*(this);
        return it;
    }

    /**
     *@brief operator -- for Iterator class
     *@return Iterator&
     */
    Iterator& operator--() {
        if (this->_index > 0) {
            this->_index--;
        }
        return *(this);
    }

    /**
     *@brief operator -- for Iterator class
     *@return Iterator&
     */
    Iterator& operator--(int) {
        Iterator it = *(this);
        --*(this);
        return it;
    }

    /**
     *@brief operator != for Iterator class
     *@return true if the elements of the array in the current index is not equal
     *@return false otherwise
     */
    bool operator!=(const Iterator& it) { return it.arr[it._index] != arr[this->_index]; }

    /**
     *@brief operator * for Iterator class
     *@return Mat1d<T, COLS> the 1-Dimensional Matrix in the current row
     */
    template <typename A, size_t CCOLS> Mat1d<A, CCOLS>& operator*() {
        Mat1d<T, COLS> mat(0);
        size_t index = 0;
        for (size_t j = this->_index * _cols; j < this->_index * _cols + _rows; ++j) {
            mat[index++] = arr[j];
        }
        return mat;
    }
};
#endif
