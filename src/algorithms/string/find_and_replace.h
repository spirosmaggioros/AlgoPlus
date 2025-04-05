#ifndef ALGOPLUS_REGEX_PATTERNS_H
#define ALGOPLUS_REGEX_PATTERNS_H

#ifdef __cplusplus
#include <regex>
#include <string>
#endif

/**
 * @brief Find and replace all occurrences of a specified regex pattern in a
 * target string with a given word.
 * @details
 * This function uses regular expressions to find all occurrences of the
 * specified regex pattern in the target string and replaces them with the given
 * word.
 * @note This function should only be used for complex patterns as it supports
 * regular expressions and might be slower for simple pattern matching and
 * replacement. Consider using find_and_replace with simple string search and
 * replace for non-complex patterns.
 * @param target The target string to search and replace within.
 * @param pattern The regex pattern to search for.
 * @param newstr The word to replace the matched patterns with.
 */
void find_and_replace_regex(std::string& target, const std::string pattern,
                            const std::string newstr) {
    std::regex regex_pattern(pattern);
    target = std::regex_replace(target, regex_pattern, newstr);
}

/**
 * @brief Finds all occurrences of a given substring in a string and replaces
 * them with a new substring.
 * @details
 * This function searches for all occurrences of the specified `oldStr` in the
 * input `str` and replaces them with the `newStr`. The replacement is done
 * in-place, meaning the original `str` is modified.
 * @note This function should be used for non-complex patterns as it does not
 * support regular expressions. For complex pattern matching and replacement,
 * consider using find_and_replace_regex that supports regular expressions.
 * @param str The string in which to find and replace substrings.
 * @param oldStr The substring to be replaced.
 * @param newStr The substring to replace `oldStr`.
 */
void find_and_replace(std::string& str, const std::string pattern, const std::string newstr) {
    std::string::size_type pos = 0u;
    while ((pos = str.find(pattern, pos)) != std::string::npos) {
        str.replace(pos, pattern.length(), newstr);
        pos += newstr.length();
    }
}

#endif // ALGOPLUS_REGEX_PATTERNS_H
