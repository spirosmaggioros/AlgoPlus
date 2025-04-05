#ifndef PALINDROME_H
#define PALINDROME_H

#ifdef __cplusplus
#include <iostream>
#include <string>
#endif

/**
 * @brief is_palindrome function. Checks if the passed string is palindrome or
 * not
 * @param str: the passed string
 * @return true if str is palindrome
 * @return false otherwise
 */
bool is_palindrome(const std::string str) {
    int64_t _size = str.size();
    for (int i = 0; i < _size / 2; i++) {
        if (str[i] != str[_size - i - 1]) {
            return false;
        }
    }
    return true;
}

#endif
