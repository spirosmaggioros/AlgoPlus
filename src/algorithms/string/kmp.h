#ifndef KMP_H
#define KMP_H

#ifdef __cplusplus
#include <iostream>
#include <string>
#include <vector>
#endif

namespace helper_array {
std::vector<int> get_array(std::string& pattern) {
    std::vector<int> failure(pattern.size() + 1);
    failure[0] = -1;
    for (int i = 0, j = -1; i < pattern.size(); i++) {
        while (j != -1 && pattern[j] != pattern[i]) {
            j = failure[j];
        }
        failure[i + 1] = ++j;
    }
    return failure;
}
}; // namespace helper_array

/**
 * @brief knuth morris pratt algorithm
 *
 * @param pattern the input pattern
 * @param text the input string
 * @return true if pattern exists in the text
 * @return false otherwise
 */
bool kmp(std::string pattern, std::string text) {
    std::vector<int> failure = helper_array::get_array(pattern);
    for (int j = 0, k = 0; j < text.size(); j++) {
        while (k != -1 && pattern[k] != text[j]) {
            k = failure[k];
        }
        if (++k == pattern.size())
            return true;
    }
    return false;
}

#endif
