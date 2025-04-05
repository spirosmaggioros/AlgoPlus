#ifndef RABIN_KARP_H
#define RABIN_KARP_H

#ifdef __cplusplus
#include <string>
#include <vector>
#endif

namespace {
const int base = 26;
const int modulus = 1e9 + 7;

/**
 * @brief Computes the hash value of a substring within a given string.
 * @param str The input string.
 * @param start The starting index of the substring.
 * @param end The ending index of the substring.
 * @return The computed hash value of the substring.
 */
size_t compute_hash(const std::string& str, size_t start, size_t end) {
    size_t curr_mod = 1;
    size_t hash_value = 0;
    for (size_t i = start; i < end; i++) {
        hash_value = (hash_value + (size_t(str[end - i - 1]) * curr_mod) % modulus) % modulus;
        curr_mod = (curr_mod * base) % modulus;
    }
    return hash_value;
}

/**
 * @brief Check if two substrings have a collision
 *
 * This function compares two substrings from two given strings to check if they
 * have a collision. A collision occurs if the characters in the corresponding
 * positions of the substrings are not equal.
 *
 * @param str1 The first string
 * @param start1 The starting position of the first substring in str1
 * @param str2 The second string
 * @param start2 The starting position of the second substring in str2
 * @param length The length of the substrings to compare
 * @return true if the substrings have a collision, false otherwise
 * @note This function assumes str1 and str2 are valid strings with lengths
 * greater than or equal to start1 + length and start2 + length respectively.
 */
bool check_collision(const std::string& str1, size_t start1, const std::string& str2, size_t start2,
                     size_t length) {
    for (size_t i = 0; i < length; ++i) {
        if (str1[start1 + i] != str2[start2 + i]) {
            return false;
        }
    }
    return true;
}
} // namespace

/**
 * @brief Executes the Rabin-Karp algorithm to search for occurrences of a
 * pattern within a text.
 *
 * @details
 * This algorithm uses rolling hash values to efficiently compare substrings of
 * the text with the pattern.
 *
 * @param text The input text to search within.
 * @param pattern The pattern to search for within the text.
 * @return A vector of starting indices of all occurrences of the pattern in the
 * text. If none were found the vector is empty.
 */
std::vector<size_t> rabin_karp(const std::string& text, const std::string& pattern) {
    std::vector<size_t> result;
    size_t pattern_length = pattern.length();
    size_t text_length = text.length();

    if (pattern_length == 0) { // if pattern is empty, it can be found at every
                               // index including the end of the text
        for (size_t i = 0; i <= text_length; i++) {
            result.push_back(i);
        }
        return result;
    }

    if (text_length < pattern_length) { // if text is shorter than pattern,
                                        // pattern can not be found
        return result;
    }

    // calculate the hash of the pattern and the hash of the first pattern_length
    // characters of the text
    size_t pattern_hash = compute_hash(pattern, 0, pattern_length);
    size_t text_hash = compute_hash(text, 0, pattern_length);

    // the highest power used in the hash calculation of the pattern
    size_t power = 1;
    for (int i = 0; i < pattern_length - 1; ++i)
        power = (power * base) % modulus;

    for (size_t i = 0; i <= text_length - pattern_length; ++i) {
        if (pattern_hash == text_hash && check_collision(text, i, pattern, 0, pattern_length)) {
            result.push_back(i);
        }

        if (i < text_length - pattern_length) {
            text_hash =
                (base * (text_hash - ((size_t)text[i] * power % modulus) + modulus) % modulus +
                 (size_t)text[i + pattern_length]) %
                modulus;
        }
    }

    return result;
}

#endif
